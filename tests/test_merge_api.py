"""片区合并供数 API 集成测试。

覆盖：发起前置条件、成功供数与守恒、无解不发布部分结果、并列确定性、
源结果切换后旧合并版 stale 且不得冒充新版、写入失败不损坏有效合并版。
"""
from __future__ import annotations

import sqlite3
import threading

import app.db as dbmod
from tests.conftest import poll_generation

OLD_V1 = {
    "items": [
        {"id": "g1", "rect": [0, 0, 10, 10], "households": 7},
        {"id": "g2", "rect": [0, 10, 10, 20], "households": 11},
    ]
}
# 一行 4 个相邻片区，每个 1x1；配合下面的旧格保证完整覆盖与可控户数
NEW_ROW4 = {
    "items": [
        {"id": "d1", "rect": [0, 0, 1, 1]},
        {"id": "d2", "rect": [1, 0, 2, 1]},
        {"id": "d3", "rect": [2, 0, 3, 1]},
        {"id": "d4", "rect": [3, 0, 4, 1]},
    ]
}
OLD_ROW4 = {
    "items": [
        {"id": f"g{i}", "rect": [i, 0, i + 1, 1], "households": h}
        for i, h in enumerate([1, 2, 1, 2])
    ]
}


def _submit(client, path, payload):
    r = client.post(path, json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def _publish_result(client, old=OLD_ROW4, new=NEW_ROW4):
    _submit(client, "/old-grids", old)
    _submit(client, "/new-districts", new)
    gen = poll_generation(client, client.post("/generations").json()["id"])
    assert gen["status"] == "succeeded"
    return gen


def poll_merge(client, job_id: int) -> dict:
    for _ in range(200):
        body = client.get(f"/merges/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        import time

        time.sleep(0.02)
    raise AssertionError("合并任务未在预期时间内结束")


def test_merge_requires_published_result(client):
    # 没有已发布结果时 409
    r = client.post("/merges", json={"min_households": 1})
    assert r.status_code == 409 and "已发布" in r.json()["detail"]
    # 非法门槛 422
    _publish_result(client)
    assert client.post("/merges", json={"min_households": 0}).status_code == 422


def test_merge_happy_path_groups_and_conservation(client):
    gen = _publish_result(client)
    # 片区户数 d1=1,d2=2,d3=1,d4=2（逐格 1x1 分摊，户数即旧格户数）
    result = client.get("/result").json()
    hh = {d["district_id"]: d["allocated_households"] for d in result["districts"]}
    assert hh == {"d1": 1, "d2": 2, "d3": 1, "d4": 2}

    r = client.post("/merges", json={"min_households": 3})
    assert r.status_code == 202
    job = poll_merge(client, r.json()["id"])
    assert job["status"] == "succeeded"
    assert job["source_generation_id"] == gen["id"]
    assert job["stale"] is False
    assert job["error"] is None

    # 最优两组：[d1,d2]=3、[d3,d4]=3（组数最多且字典序最小）
    assert job["groups"] == [
        {"group_seq": 0, "households": 3, "district_ids": ["d1", "d2"]},
        {"group_seq": 1, "households": 3, "district_ids": ["d3", "d4"]},
    ]
    # 组户数守恒于源结果总户数 6
    assert sum(g["households"] for g in job["groups"]) == 6

    # 当前供数视图
    supply = client.get("/merge-result")
    assert supply.status_code == 200
    body = supply.json()
    assert body["id"] == job["id"]
    assert body["stale"] is False
    assert body["total_households"] == 6
    assert body["source_generation_id"] == gen["id"]

    # 任务列表
    listed = client.get("/merges").json()
    assert [j["id"] for j in listed] == [job["id"]]
    assert client.get("/merges/99").status_code == 404
    assert client.get("/merge-result").status_code == 200


def test_merge_infeasible_publishes_nothing(client):
    _publish_result(client)
    # 总量 6，门槛 7 -> 无解
    r = client.post("/merges", json={"min_households": 7})
    job = poll_merge(client, r.json()["id"])
    assert job["status"] == "failed"
    assert job["groups"] is None
    assert "低于最低户数" in job["error"]

    # 无任何供数结果 -> 404，没有发布部分结果
    assert client.get("/merge-result").status_code == 404

    # 角点相接场景：两个不相邻片区各 3 户、门槛 5 -> 分量各自不达标
    old_corner = {
        "items": [
            {"id": "g1", "rect": [0, 0, 1, 1], "households": 3},
            {"id": "g2", "rect": [1, 1, 2, 2], "households": 3},
        ]
    }
    new_corner = {
        "items": [
            {"id": "e1", "rect": [0, 0, 1, 1]},
            {"id": "e2", "rect": [1, 1, 2, 2]},
        ]
    }
    _submit(client, "/old-grids", old_corner)
    _submit(client, "/new-districts", new_corner)
    gen2 = poll_generation(client, client.post("/generations").json()["id"])
    assert gen2["status"] == "succeeded"
    job2 = poll_merge(client, client.post("/merges", json={"min_households": 5}).json()["id"])
    assert job2["status"] == "failed"
    assert "e1" in job2["error"] and "e2" in job2["error"]
    assert client.get("/merge-result").status_code == 404


def test_merge_tie_break_is_deterministic(client):
    # 每户 1 的 2x2 环，门槛 2：两种两组划分，取字典序最小
    old_sq = {
        "items": [
            {"id": "g1", "rect": [0, 1, 1, 2], "households": 1},
            {"id": "g2", "rect": [1, 1, 2, 2], "households": 1},
            {"id": "g3", "rect": [0, 0, 1, 1], "households": 1},
            {"id": "g4", "rect": [1, 0, 2, 1], "households": 1},
        ]
    }
    new_sq = {
        "items": [
            {"id": "d1", "rect": [0, 1, 1, 2]},
            {"id": "d2", "rect": [1, 1, 2, 2]},
            {"id": "d3", "rect": [0, 0, 1, 1]},
            {"id": "d4", "rect": [1, 0, 2, 1]},
        ]
    }
    _publish_result(client, old=old_sq, new=new_sq)
    job = poll_merge(client, client.post("/merges", json={"min_households": 2}).json()["id"])
    assert job["status"] == "succeeded"
    assert job["groups"][0]["district_ids"] == ["d1", "d2"]
    assert job["groups"][1]["district_ids"] == ["d3", "d4"]


def test_source_switch_marks_old_merge_stale_and_recompute_serves_new(client):
    gen1 = _publish_result(client)
    job1 = poll_merge(client, client.post("/merges", json={"min_households": 3}).json()["id"])
    assert job1["status"] == "succeeded" and job1["stale"] is False

    # 发布一份全新的生成结果（源结果更替）
    _submit(client, "/old-grids", {
        "items": [{"id": "h1", "rect": [0, 0, 4, 1], "households": 9}]
    })
    _submit(client, "/new-districts", {
        "items": [{"id": "e1", "rect": [0, 0, 4, 1]}]
    })
    gen2 = poll_generation(client, client.post("/generations").json()["id"])
    assert gen2["status"] == "succeeded" and gen2["id"] != gen1["id"]

    # 旧合并任务与当前供数都被标记 stale，但内容保留、不得冒充新版
    old_job = client.get(f"/merges/{job1['id']}").json()
    assert old_job["stale"] is True
    assert old_job["source_generation_id"] == gen1["id"]
    supply = client.get("/merge-result").json()
    assert supply["stale"] is True
    assert supply["id"] == job1["id"]

    # 对新源结果重算成功后才供应新版
    job2 = poll_merge(client, client.post("/merges", json={"min_households": 1}).json()["id"])
    assert job2["status"] == "succeeded"
    assert job2["source_generation_id"] == gen2["id"]
    assert job2["stale"] is False
    new_supply = client.get("/merge-result").json()
    assert new_supply["id"] == job2["id"]
    assert new_supply["stale"] is False
    assert [[g["district_ids"]] for g in new_supply["groups"]] == [[["e1"]]]
    assert new_supply["total_households"] == 9
    # 旧任务仍可查且仍标记 stale
    assert client.get(f"/merges/{job1['id']}").json()["stale"] is True


def test_merge_input_pinned_to_source_when_switch_happens_during_run(client):
    """合并计算期间源结果的输入被替换：任务仍按绑定的旧源计算。

    工作池单线程串行，因此这里在合并阻塞期间提交新的 old/new 版本
    （版本提交不经过工作池）；待合并完成后再触发生成切换源结果，
    合并版应被如实标记 stale 而不是冒充新版。
    """
    gen1 = _publish_result(client)

    started = threading.Event()
    release = threading.Event()
    original_load = dbmod._load_merge_input

    def blocked_load(conn, source_generation_id):
        started.set()
        assert release.wait(timeout=5)
        return original_load(conn, source_generation_id)

    dbmod._load_merge_input = blocked_load
    try:
        r = client.post("/merges", json={"min_households": 3})
        job_id = r.json()["id"]
        assert started.wait(timeout=5)

        # 计算进行中替换两份输入（产生新版本，但尚未生成、未切换源结果）
        _submit(client, "/old-grids", {
            "items": [{"id": "h1", "rect": [0, 0, 4, 1], "households": 9}]
        })
        new_v = _submit(client, "/new-districts", {
            "items": [{"id": "e1", "rect": [0, 0, 4, 1]}]
        })
        release.set()
        job = poll_merge(client, job_id)
    finally:
        dbmod._load_merge_input = original_load
        release.set()

    assert job["status"] == "succeeded"
    # 输入固定为旧源（d1..d4），没有被尚未发布的新输入（e1）污染
    assert job["source_generation_id"] == gen1["id"]
    flat = [x for g in job["groups"] for x in g["district_ids"]]
    assert sorted(flat) == ["d1", "d2", "d3", "d4"]
    assert client.get("/merge-result").json()["stale"] is False

    # 现在触发生成，源结果正式切换
    gen2 = poll_generation(client, client.post("/generations").json()["id"])
    assert gen2["status"] == "succeeded" and gen2["id"] != gen1["id"]
    assert gen2["new_version"] == new_v["version"]
    stale_job = client.get(f"/merges/{job_id}").json()
    assert stale_job["stale"] is True
    assert client.get("/merge-result").json()["stale"] is True


def test_write_failure_keeps_previous_valid_merge(client):
    # 先成功发布一版合并供数
    _publish_result(client)
    job1 = poll_merge(client, client.post("/merges", json={"min_households": 3}).json()["id"])
    assert job1["status"] == "succeeded"
    good = client.get("/merge-result").json()

    # 让下一个后台合并连接在发布事务写 merge_groups 时失败：
    # 事务整体回滚，current_merge 不被触碰，旧合并版保持有效。
    class FailingConn:
        """连接代理：只拦截 merge_groups 插入，其余属性/方法委托真实连接。"""

        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def execute(self, sql, params=()):
            if sql.startswith("INSERT INTO merge_groups"):
                raise sqlite3.OperationalError("injected write failure")
            return self._real.execute(sql, params)

        def __enter__(self):
            self._real.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._real.__exit__(exc_type, exc, tb)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_real"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_real"), name, value)

    def flaky_hook(conn):
        return FailingConn(conn)

    dbmod.connection_hook = flaky_hook
    try:
        job2 = poll_merge(client, client.post("/merges", json={"min_households": 2}).json()["id"])
    finally:
        dbmod.connection_hook = None

    assert job2["status"] == "failed"
    assert "injected write failure" in (job2["error"] or "")
    # 仍有效的旧合并版未被损坏
    still = client.get("/merge-result").json()
    assert still["id"] == good["id"]
    assert still["groups"] == good["groups"]
    assert still["total_households"] == good["total_households"]

    # 故障消除后重算成功，正常切换
    job3 = poll_merge(client, client.post("/merges", json={"min_households": 2}).json()["id"])
    assert job3["status"] == "succeeded"
    assert client.get("/merge-result").json()["id"] == job3["id"]


def test_failed_merge_does_not_block_later_success(client):
    _publish_result(client)
    bad = poll_merge(client, client.post("/merges", json={"min_households": 100}).json()["id"])
    assert bad["status"] == "failed"
    ok = poll_merge(client, client.post("/merges", json={"min_households": 1}).json()["id"])
    assert ok["status"] == "succeeded"
    # 门槛 1 时每个片区自身达标 -> 不合并，4 个单成员组
    assert [g["district_ids"] for g in ok["groups"]] == [["d1"], ["d2"], ["d3"], ["d4"]]
    assert client.get("/merge-result").json()["id"] == ok["id"]


def test_merge_endpoints_present_in_openapi(client):
    spec = client.get("/openapi.json").json()
    for path in ("/merges", "/merges/{job_id}", "/merge-result"):
        assert path in spec["paths"]
