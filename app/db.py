"""SQLite 持久化层。

设计要点：
- 输入按“整版提交”存为不可变版本（versions + cells），替换即产生新版本。
- 生成任务在创建（生成开始）的同一事务内固定 old/new 版本号，
  之后即使输入被替换，任务仍按该快照计算。
- current_result 为单行表，只在新结果完整计算成功后于同一事务内切换，
  失败或进行中绝不触碰它，旧发布结果保持可查。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import allocation as alloc
from . import merging as mrg
from .config import DB_PATH, MAX_ITEMS_PER_SUBMISSION

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL CHECK (kind IN ('old', 'new')),
    created_at  TEXT NOT NULL,
    item_count  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS cells (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id   INTEGER NOT NULL REFERENCES versions(id),
    seq          INTEGER NOT NULL,
    cell_id      TEXT NOT NULL,
    x1           INTEGER NOT NULL,
    y1           INTEGER NOT NULL,
    x2           INTEGER NOT NULL,
    y2           INTEGER NOT NULL,
    area         INTEGER NOT NULL,
    households   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_cells_version ON cells(version_id);

CREATE TABLE IF NOT EXISTS generations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    status        TEXT NOT NULL CHECK (status IN ('pending', 'running', 'succeeded', 'failed')),
    old_version   INTEGER REFERENCES versions(id),
    new_version   INTEGER REFERENCES versions(id),
    error         TEXT,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT
);

CREATE TABLE IF NOT EXISTS result_parts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id  INTEGER NOT NULL REFERENCES generations(id),
    district_id    TEXT NOT NULL,
    old_id         TEXT NOT NULL,
    overlap_area   INTEGER NOT NULL,
    households     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_parts_generation ON result_parts(generation_id);

-- 单行表：始终指向“当前已发布结果”
CREATE TABLE IF NOT EXISTS current_result (
    only_row       INTEGER PRIMARY KEY CHECK (only_row = 1),
    generation_id  INTEGER NOT NULL REFERENCES generations(id),
    published_at   TEXT NOT NULL,
    total_households INTEGER NOT NULL
);

-- 片区合并供数任务：绑定创建时刻的已发布生成任务（源结果）与最低户数门槛
CREATE TABLE IF NOT EXISTS merge_jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    status          TEXT NOT NULL CHECK (status IN ('pending', 'running', 'succeeded', 'failed')),
    source_generation_id INTEGER NOT NULL REFERENCES generations(id),
    min_households  INTEGER NOT NULL CHECK (min_households >= 1),
    error           TEXT,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT
);

-- 合并任务产出的组：仅对成功任务写入，随原子发布事务一起落库
CREATE TABLE IF NOT EXISTS merge_groups (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    merge_job_id    INTEGER NOT NULL REFERENCES merge_jobs(id),
    group_seq       INTEGER NOT NULL,
    district_id     TEXT NOT NULL,
    member_seq      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_merge_groups_job ON merge_groups(merge_job_id);

-- 单行表：始终指向“当前供数的合并版”；只有整版写入成功才会切换
CREATE TABLE IF NOT EXISTS current_merge (
    only_row       INTEGER PRIMARY KEY CHECK (only_row = 1),
    merge_job_id   INTEGER NOT NULL REFERENCES merge_jobs(id),
    published_at   TEXT NOT NULL,
    total_households INTEGER NOT NULL
);
"""


class NoVersionError(Exception):
    """缺少某一类输入的已提交版本。"""


class ValidationError(Exception):
    """输入业务校验失败（编号重复、内部重叠等）。"""


class NoResultError(Exception):
    """尚无已发布的生成结果，无法作为合并供数的输入。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 测试钩子：非 None 时，后台任务新建的每个连接都会经过它（可用于注入故障）。
# 生产路径始终为 None，不改变任何行为。
connection_hook = None


def get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=30000;")
    # 保证任何连接（含后台工作线程）打开时表结构都已就绪（DDL 幂等）
    conn.executescript(SCHEMA)
    if connection_hook is not None:
        # 测试钩子可返回包装连接（如故障注入代理）；生产钩子为 None
        hooked = connection_hook(conn)
        if hooked is not None:
            return hooked
    return conn


def init_db(db_path: Path | None = None) -> None:
    conn = get_conn(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ---------- 输入版本 ----------

def _validate_items(kind: str, items: list[dict[str, Any]]) -> None:
    """业务校验：数量上限、编号唯一、同版矩形互不重叠。"""
    if not items:
        raise ValidationError("提交内容不能为空")
    if len(items) > MAX_ITEMS_PER_SUBMISSION:
        raise ValidationError(
            f"每版最多 {MAX_ITEMS_PER_SUBMISSION} 个条目，收到 {len(items)} 个"
        )
    ids = [it["id"] for it in items]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValidationError(f"条目编号必须唯一，重复编号：{', '.join(dupes)}")
    rects = [(it["id"], it["rect"]) for it in items]
    try:
        alloc.assert_no_internal_overlap(rects)  # 抛 ValueError
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def insert_version(kind: str, items: list[dict[str, Any]], conn: sqlite3.Connection) -> int:
    """在给定连接/事务内写入一个不可变版本，返回版本号。"""
    _validate_items(kind, items)
    cur = conn.execute(
        "INSERT INTO versions(kind, created_at, item_count) VALUES (?, ?, ?)",
        (kind, _now(), len(items)),
    )
    version_id = cur.lastrowid
    for seq, it in enumerate(items):
        x1, y1, x2, y2 = it["rect"]
        conn.execute(
            "INSERT INTO cells(version_id, seq, cell_id, x1, y1, x2, y2, area, households)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                version_id,
                seq,
                it["id"],
                x1,
                y1,
                x2,
                y2,
                alloc.rect_area(it["rect"]),
                it.get("households"),
            ),
        )
    return version_id


def latest_version_id(kind: str, conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT MAX(id) AS m FROM versions WHERE kind = ?", (kind,)
    ).fetchone()
    return row["m"]


def _load_version(conn: sqlite3.Connection, version_id: int) -> dict[str, Any]:
    vrow = conn.execute("SELECT * FROM versions WHERE id = ?", (version_id,)).fetchone()
    if vrow is None:
        raise NoVersionError(f"版本 {version_id} 不存在")
    cells = conn.execute(
        "SELECT * FROM cells WHERE version_id = ? ORDER BY seq", (version_id,)
    ).fetchall()
    return {
        "version": vrow["id"],
        "kind": vrow["kind"],
        "created_at": vrow["created_at"],
        "item_count": vrow["item_count"],
        "items": [
            {
                "id": c["cell_id"],
                "x1": c["x1"],
                "y1": c["y1"],
                "x2": c["x2"],
                "y2": c["y2"],
                "area": c["area"],
                "households": c["households"],
            }
            for c in cells
        ],
    }


def get_version(version_id: int, conn: sqlite3.Connection) -> dict[str, Any]:
    return _load_version(conn, version_id)


def list_versions(kind: str | None, conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if kind:
        rows = conn.execute(
            "SELECT id FROM versions WHERE kind = ? ORDER BY id", (kind,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT id FROM versions ORDER BY id").fetchall()
    return [_load_version(conn, r["id"]) for r in rows]


# ---------- 生成任务 ----------

def create_generation(conn: sqlite3.Connection) -> int:
    """创建生成任务并在同一事务固定当前最新 old/new 版本快照。"""
    old_v = latest_version_id("old", conn)
    new_v = latest_version_id("new", conn)
    missing = []
    if old_v is None:
        missing.append("旧网格")
    if new_v is None:
        missing.append("新片区")
    if missing:
        raise NoVersionError("缺少已提交的" + "、".join(missing) + "版本，无法生成")
    cur = conn.execute(
        "INSERT INTO generations(status, old_version, new_version, created_at)"
        " VALUES ('pending', ?, ?, ?)",
        (old_v, new_v, _now()),
    )
    return cur.lastrowid


def _load_snapshot(conn: sqlite3.Connection, gen_id: int) -> dict[str, Any]:
    grow = conn.execute("SELECT * FROM generations WHERE id = ?", (gen_id,)).fetchone()
    old_version = _load_version(conn, grow["old_version"])
    new_version = _load_version(conn, grow["new_version"])
    old_cells = [
        alloc.OldCell(
            id=c["id"],
            rect=(c["x1"], c["y1"], c["x2"], c["y2"]),
            households=c["households"],
        )
        for c in old_version["items"]
    ]
    districts = [
        alloc.NewDistrict(
            id=c["id"], rect=(c["x1"], c["y1"], c["x2"], c["y2"])
        )
        for c in new_version["items"]
    ]
    return {
        "row": grow,
        "old_version": old_version,
        "new_version": new_version,
        "old_cells": old_cells,
        "districts": districts,
    }


def run_generation(gen_id: int, db_path: Path | None = None) -> None:
    """工作线程入口：按任务固定的快照计算并原子发布；失败只标记任务。

    任何异常都会把任务落为 failed（而不是永久 running），
    且失败路径绝不写入 current_result，旧发布结果保持可查。
    """
    conn = get_conn(db_path)
    try:
        # 1. 抢占任务并读取快照（任务版本号在创建时已固定，不受新提交影响）
        conn.execute(
            "UPDATE generations SET status='running', started_at=? "
            "WHERE id=? AND status='pending'",
            (_now(), gen_id),
        )
        conn.commit()
        snap = _load_snapshot(conn, gen_id)

        # 2. 纯函数计算（覆盖校验 + 最大余数分摊）
        try:
            allocations = alloc.allocate(snap["old_cells"], snap["districts"])
        except ValueError as exc:
            _mark_failed(conn, gen_id, str(exc))
            return

        total = sum(c.households for c in snap["old_cells"])
        allocated_total = sum(a.households for a in allocations)
        if allocated_total != total:
            # 理论上不会发生；作为守恒护栏，宁可失败也不发布错误结果
            _mark_failed(
                conn,
                gen_id,
                f"守恒校验失败：原总户数 {total}，分摊合计 {allocated_total}",
            )
            return

        # 3. 原子发布：明细 + 任务成功 + 切换当前结果，同一事务
        # 事务异常自动回滚，current_result 保持旧值
        with conn:
            for a in allocations:
                conn.execute(
                    "INSERT INTO result_parts(generation_id, district_id, old_id,"
                    " overlap_area, households) VALUES (?, ?, ?, ?, ?)",
                    (gen_id, a.district_id, a.old_id, a.overlap_area, a.households),
                )
            conn.execute(
                "UPDATE generations SET status='succeeded', error=NULL, finished_at=? WHERE id=?",
                (_now(), gen_id),
            )
            conn.execute(
                "INSERT INTO current_result(only_row, generation_id, published_at,"
                " total_households) VALUES (1, ?, ?, ?) "
                "ON CONFLICT(only_row) DO UPDATE SET "
                "generation_id=excluded.generation_id,"
                " published_at=excluded.published_at,"
                " total_households=excluded.total_households",
                (gen_id, _now(), total),
            )
    except Exception as exc:  # 兜底：任何意外都落 failed，不卡死任务
        try:
            _mark_failed(conn, gen_id, f"生成失败：{type(exc).__name__}: {exc}")
        except sqlite3.Error:
            pass
    finally:
        conn.close()


def _mark_failed(conn: sqlite3.Connection, gen_id: int, message: str) -> None:
    conn.execute(
        "UPDATE generations SET status='failed', error=?, finished_at=? WHERE id=?",
        (message, _now(), gen_id),
    )
    conn.commit()


def get_generation(gen_id: int, conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM generations WHERE id = ?", (gen_id,)).fetchone()
    return dict(row) if row else None


def list_generations(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM generations ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def get_current_result(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """组装当前发布结果：片区汇总 + 两份输入版本。无发布时返回 None。"""
    cur = conn.execute("SELECT * FROM current_result WHERE only_row=1").fetchone()
    if cur is None:
        return None
    gen_id = cur["generation_id"]
    grow = conn.execute("SELECT * FROM generations WHERE id=?", (gen_id,)).fetchone()
    old_version = _load_version(conn, grow["old_version"])
    new_version = _load_version(conn, grow["new_version"])

    parts = conn.execute(
        "SELECT * FROM result_parts WHERE generation_id=? ORDER BY id", (gen_id,)
    ).fetchall()
    by_district: dict[str, list[dict[str, Any]]] = {
        c["id"]: [] for c in new_version["items"]
    }
    for p in parts:
        by_district.setdefault(p["district_id"], []).append(
            {
                "district_id": p["district_id"],
                "old_id": p["old_id"],
                "overlap_area": p["overlap_area"],
                "households": p["households"],
            }
        )
    districts = []
    for d_id in sorted(by_district, key=alloc._id_sort_key):
        detail = sorted(by_district[d_id], key=lambda a: alloc._id_sort_key(a["old_id"]))
        districts.append(
            {
                "district_id": d_id,
                "allocated_households": sum(a["households"] for a in detail),
                "detail": detail,
            }
        )
    return {
        "generation_id": gen_id,
        "published_at": cur["published_at"],
        "total_households": cur["total_households"],
        "old_version": old_version,
        "new_version": new_version,
        "districts": districts,
    }


# ---------- 片区合并供数 ----------

def create_merge_job(min_households: int, conn: sqlite3.Connection) -> int:
    """创建合并任务，在同一事务内绑定“当前已发布生成结果”作为唯一输入。

    源结果（generation_id）在此刻固定；此后即使新的生成结果发布，
    本任务仍按绑定的源结果计算，避免把旧合并版冒充为新版。
    """
    if min_households < 1:
        raise ValidationError("最低户数必须为正整数")
    cur = conn.execute("SELECT generation_id FROM current_result WHERE only_row=1").fetchone()
    if cur is None:
        raise NoResultError("尚无已发布的生成结果，无法发起片区合并")
    row = conn.execute(
        "INSERT INTO merge_jobs(status, source_generation_id, min_households, created_at)"
        " VALUES ('pending', ?, ?, ?)",
        (cur["generation_id"], min_households, _now()),
    )
    return row.lastrowid


def _load_merge_input(
    conn: sqlite3.Connection, source_generation_id: int
) -> list[mrg.MergeDistrict]:
    """从绑定的源生成结果读取片区矩形与已发布户数（合并任务的唯一数据来源）。"""
    grow = conn.execute(
        "SELECT * FROM generations WHERE id=?", (source_generation_id,)
    ).fetchone()
    if grow is None or grow["status"] != "succeeded":
        raise NoResultError(f"生成任务 {source_generation_id} 不是可用的已发布结果")
    new_version = _load_version(conn, grow["new_version"])
    # 以已发布的 result_parts 汇总户数；无分摊明细的片区户数为 0
    totals: dict[str, int] = {}
    for cell in new_version["items"]:
        totals[cell["id"]] = 0
    parts = conn.execute(
        "SELECT district_id, SUM(households) AS h FROM result_parts"
        " WHERE generation_id=? GROUP BY district_id",
        (source_generation_id,),
    ).fetchall()
    for p in parts:
        totals[p["district_id"]] = p["h"]
    return [
        mrg.MergeDistrict(
            id=cell["id"],
            rect=(cell["x1"], cell["y1"], cell["x2"], cell["y2"]),
            households=totals[cell["id"]],
        )
        for cell in new_version["items"]
    ]


def run_merge_job(merge_job_id: int, db_path: Path | None = None) -> None:
    """工作线程入口：按任务绑定的源结果计算合并并原子发布。

    - 无解（MergeInfeasibleError）只把任务落为 failed 并写明原因，不发布部分结果；
    - 任何写入失败随事务整体回滚，current_merge 仍指向上一份有效合并版；
    - 源结果在此期间被切换也不影响本任务（输入按绑定的 source_generation_id 读取）。
    """
    conn = get_conn(db_path)
    try:
        conn.execute(
            "UPDATE merge_jobs SET status='running', started_at=? "
            "WHERE id=? AND status='pending'",
            (_now(), merge_job_id),
        )
        conn.commit()
        job = conn.execute(
            "SELECT * FROM merge_jobs WHERE id=?", (merge_job_id,)
        ).fetchone()

        # 输入完全来自绑定的源生成任务，不读取 current_result 指针
        districts = _load_merge_input(conn, job["source_generation_id"])

        try:
            groups = mrg.merge_districts(districts, job["min_households"])
        except mrg.MergeInfeasibleError as exc:
            _mark_merge_failed(conn, merge_job_id, str(exc))
            return
        except ValueError as exc:
            _mark_merge_failed(conn, merge_job_id, str(exc))
            return

        # 守恒护栏：组内户数合计必须等于源结果总户数
        total = sum(d.households for d in districts)
        grouped_total = 0
        for group in groups:
            grouped_total += sum(
                d.households for d in districts if d.id in group
            )
        if grouped_total != total:
            _mark_merge_failed(
                conn,
                merge_job_id,
                f"合并守恒校验失败：源总户数 {total}，合并合计 {grouped_total}",
            )
            return

        # 原子发布：组明细 + 任务成功 + 切换当前供数指针，同一事务。
        # 事务异常自动回滚，current_merge 保持旧值，仍有效的旧合并版不受损。
        with conn:
            for gseq, group in enumerate(groups):
                for mseq, district_id in enumerate(group):
                    conn.execute(
                        "INSERT INTO merge_groups(merge_job_id, group_seq, district_id,"
                        " member_seq) VALUES (?, ?, ?, ?)",
                        (merge_job_id, gseq, district_id, mseq),
                    )
            conn.execute(
                "UPDATE merge_jobs SET status='succeeded', error=NULL, finished_at=?"
                " WHERE id=?",
                (_now(), merge_job_id),
            )
            conn.execute(
                "INSERT INTO current_merge(only_row, merge_job_id, published_at,"
                " total_households) VALUES (1, ?, ?, ?) "
                "ON CONFLICT(only_row) DO UPDATE SET "
                "merge_job_id=excluded.merge_job_id,"
                " published_at=excluded.published_at,"
                " total_households=excluded.total_households",
                (merge_job_id, _now(), total),
            )
    except Exception as exc:  # 兜底：任何意外都落 failed，不卡死任务且不动当前供数
        try:
            _mark_merge_failed(
                conn, merge_job_id, f"合并失败：{type(exc).__name__}: {exc}"
            )
        except sqlite3.Error:
            pass
    finally:
        conn.close()


def _mark_merge_failed(conn: sqlite3.Connection, job_id: int, message: str) -> None:
    conn.execute(
        "UPDATE merge_jobs SET status='failed', error=?, finished_at=? WHERE id=?",
        (message, _now(), job_id),
    )
    conn.commit()


def get_merge_job(merge_job_id: int, conn: sqlite3.Connection) -> dict[str, Any] | None:
    """组装合并任务视图；stale 表示源结果已被新发布结果替换（旧合并版不得当新版供应）。"""
    row = conn.execute("SELECT * FROM merge_jobs WHERE id=?", (merge_job_id,)).fetchone()
    if row is None:
        return None
    return _build_merge_view(conn, row)


def list_merge_jobs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM merge_jobs ORDER BY id").fetchall()
    return [_build_merge_view(conn, r) for r in rows]


def _current_generation_id(conn: sqlite3.Connection) -> int | None:
    cur = conn.execute(
        "SELECT generation_id FROM current_result WHERE only_row=1"
    ).fetchone()
    return None if cur is None else cur["generation_id"]


def _build_merge_view(
    conn: sqlite3.Connection, row: sqlite3.Row
) -> dict[str, Any]:
    current_gen = _current_generation_id(conn)
    groups: list[dict[str, Any]] | None = None
    if row["status"] == "succeeded":
        group_rows = conn.execute(
            "SELECT * FROM merge_groups WHERE merge_job_id=? ORDER BY group_seq, member_seq",
            (row["id"],),
        ).fetchall()
        groups = []
        current_seq = -1
        for gr in group_rows:
            if gr["group_seq"] != current_seq:
                groups.append(
                    {
                        "group_seq": gr["group_seq"],
                        "households": 0,
                        "district_ids": [],
                    }
                )
                current_seq = gr["group_seq"]
            groups[-1]["district_ids"].append(gr["district_id"])
        # 户数按源结果汇总补齐，保证视图自身可核对守恒
        totals = dict(
            (p["district_id"], p["h"])
            for p in conn.execute(
                "SELECT district_id, SUM(households) AS h FROM result_parts"
                " WHERE generation_id=? GROUP BY district_id",
                (row["source_generation_id"],),
            ).fetchall()
        )
        for g in groups:
            g["households"] = sum(totals.get(d, 0) for d in g["district_ids"])
    return {
        "id": row["id"],
        "status": row["status"],
        "source_generation_id": row["source_generation_id"],
        "min_households": row["min_households"],
        "stale": current_gen is not None
        and current_gen != row["source_generation_id"],
        "error": row["error"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "groups": groups,
    }


def get_current_merge(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """当前供数的合并版（无则 None）；stale=1 时表示其源结果已被替换。"""
    cur = conn.execute("SELECT * FROM current_merge WHERE only_row=1").fetchone()
    if cur is None:
        return None
    row = conn.execute(
        "SELECT * FROM merge_jobs WHERE id=?", (cur["merge_job_id"],)
    ).fetchone()
    view = _build_merge_view(conn, row)
    view["published_at"] = cur["published_at"]
    view["total_households"] = cur["total_households"]
    return view
