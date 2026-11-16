"""片区合并纯逻辑单元测试：共边判定、连通分量、最优化、无解与守恒。"""
from __future__ import annotations

import random

import pytest

from app import merging as mrg
from app.merging import (
    MergeDistrict,
    MergeInfeasibleError,
    build_adjacency,
    connected_components,
    merge_districts,
    share_edge,
)


def D(did: str, rect, h: int) -> MergeDistrict:
    return MergeDistrict(did, rect, h)


# ---------- 共边几何 ----------

def test_share_edge_basic_orientations():
    # 右-左边相接，公共边长度为正
    assert share_edge((0, 0, 1, 1), (1, 0, 2, 1)) is True
    # 上-下边相接
    assert share_edge((0, 0, 1, 1), (0, 1, 1, 2)) is True
    # 部分边相接（投影只重叠一半）也算共边
    assert share_edge((0, 0, 2, 2), (2, 1, 3, 3)) is True
    assert share_edge((0, 0, 2, 2), (1, 2, 3, 3)) is True


def test_corner_touch_is_not_adjacent():
    # 仅角点相接：两种对角方向都不算相邻
    assert share_edge((0, 0, 1, 1), (1, 1, 2, 2)) is False
    assert share_edge((1, 1, 2, 2), (0, 0, 1, 1)) is False
    # 相离
    assert share_edge((0, 0, 1, 1), (2, 2, 3, 3)) is False
    # 边在同一直线但投影退化为点（T 型角点）也不算
    assert share_edge((0, 0, 1, 1), (1, 1, 2, 3)) is False


def test_adjacency_and_components():
    # d1 与 d2 共边；d3 只与 d2 角点相接 -> 两个连通分量
    districts = [
        D("d1", (0, 0, 1, 1), 5),
        D("d2", (1, 0, 2, 1), 5),
        D("d3", (2, 1, 3, 2), 5),
    ]
    adj = build_adjacency(districts)
    assert adj[0] == 0b010
    assert adj[1] == 0b001
    assert adj[2] == 0
    comps = connected_components(3, adj)
    assert comps == [(0, 1), (2,)]


# ---------- 无解 ----------

def test_infeasible_when_component_below_threshold():
    # 两个互不相邻的单片区，各 3 户；门槛 5 -> 两个分量都无法达标
    districts = [
        D("d1", (0, 0, 1, 1), 3),
        D("d2", (5, 5, 6, 6), 3),
    ]
    with pytest.raises(MergeInfeasibleError, match="低于最低户数") as exc:
        merge_districts(districts, 5)
    # 错误信息应点名涉及的片区与门槛，便于统计员定位
    assert "d1" in str(exc.value) and "d2" in str(exc.value)


def test_infeasible_corner_components_cannot_merge():
    # 角点相接的两个片区不能并成一组：各 3 户、门槛 5 时无解
    districts = [
        D("d1", (0, 0, 1, 1), 3),
        D("d2", (1, 1, 2, 2), 3),
    ]
    with pytest.raises(MergeInfeasibleError):
        merge_districts(districts, 5)


def test_empty_input_rejected():
    with pytest.raises(MergeInfeasibleError):
        merge_districts([], 1)
    with pytest.raises(ValueError):
        merge_districts([D("d1", (0, 0, 1, 1), 1)], 0)


# ---------- 组数最大化 ----------

def test_max_group_count_preferred_on_chain():
    # 链 d1-d2-d3-d4，各 1 户，门槛 2：最多 2 组
    districts = [D(f"d{i}", (i, 0, i + 1, 1), 1) for i in range(1, 5)]
    groups = merge_districts(districts, 2)
    assert groups == [["d1", "d2"], ["d3", "d4"]]
    assert len(groups) == 2


def test_max_group_count_with_unequal_households():
    # 链 5 个片区，户数 [3, 1, 3, 1, 3]，门槛 4：
    # 两组方案存在（如 d1+d2=4, d3+d4+d5=7）；单组方案虽可行也必须选两组
    districts = [
        D("d1", (0, 0, 1, 1), 3),
        D("d2", (1, 0, 2, 1), 1),
        D("d3", (2, 0, 3, 1), 3),
        D("d4", (3, 0, 4, 1), 1),
        D("d5", (4, 0, 5, 1), 3),
    ]
    groups = merge_districts(districts, 4)
    assert len(groups) == 2
    # 全部片区恰好出现一次
    flat = [x for g in groups for x in g]
    assert sorted(flat) == ["d1", "d2", "d3", "d4", "d5"]
    # 每组达标
    hh = {d.id: d.households for d in districts}
    assert all(sum(hh[x] for x in g) >= 4 for g in groups)


def test_singleton_groups_when_every_district_meets_threshold():
    # 每个片区自身达标时，组数最多的方案就是不合并
    districts = [D(f"d{i}", (i, 0, i + 1, 1), 10) for i in range(1, 4)]
    groups = merge_districts(districts, 5)
    assert groups == [["d1"], ["d2"], ["d3"]]


# ---------- 并列的确定性 ----------

def test_tie_break_lexicographic_on_path_alternative_splits():
    # 链 d1-d2-d3-d4-d5，户数 [2,2,2,2,2]，门槛 4：
    # 两组的可行切分多种（相邻对 + 三元组），取字典序最小划分。
    districts = [D(f"d{i}", (i, 0, i + 1, 1), 2) for i in range(1, 6)]
    groups = merge_districts(districts, 4)
    assert len(groups) == 2
    # 可行划分有 [d1d2 | d3d4d5]、[d1d2d3 | d4d5] 等；
    # 字典序最小者以最小编号开始的最小组开头 -> ['d1','d2']
    assert groups == [["d1", "d2"], ["d3", "d4", "d5"]]


def test_tie_break_on_cycle_picks_lexicographic_partition():
    # 2x2 环（四角片区，相邻含上下左右四条共边；对角仅角点相接）。
    # 每户 1、门槛 2：最优两组，存在两种相邻对划分，取字典序最小。
    districts = [
        D("d1", (0, 1, 1, 2), 1),  # 左上
        D("d2", (1, 1, 2, 2), 1),  # 右上
        D("d3", (0, 0, 1, 1), 1),  # 左下
        D("d4", (1, 0, 2, 1), 1),  # 右下
    ]
    groups = merge_districts(districts, 2)
    assert len(groups) == 2
    flat = sorted(x for g in groups for x in g)
    assert flat == ["d1", "d2", "d3", "d4"]
    # 两种两组划分：(d1,d2)+(d3,d4) 与 (d1,d3)+(d2,d4)；前者字典序更小
    assert groups == [["d1", "d2"], ["d3", "d4"]]
    # 对角片区不得同组（角点相接不相邻）
    for g in groups:
        assert not ({"d1", "d4"} <= set(g))
        assert not ({"d2", "d3"} <= set(g))


def test_tie_break_numeric_id_ordering():
    # 编号 "2"/"10" 按数值排序而非字符串：2 在 10 前
    districts = [
        D("10", (0, 0, 1, 1), 1),
        D("2", (1, 0, 2, 1), 1),
    ]
    groups = merge_districts(districts, 2)
    assert groups == [["2", "10"]]


def test_string_id_lexicographic_ordering():
    districts = [
        D("b", (0, 0, 1, 1), 1),
        D("a", (1, 0, 2, 1), 1),
    ]
    groups = merge_districts(districts, 2)
    assert groups == [["a", "b"]]


# ---------- 守恒与零户数 ----------

def test_conservation_with_zero_household_district():
    # 零户数片区必须依附相邻片区才能进入达标组；总户数守恒
    districts = [
        D("d1", (0, 0, 1, 1), 0),
        D("d2", (1, 0, 2, 1), 5),
    ]
    groups = merge_districts(districts, 1)
    assert groups == [["d1", "d2"]]


def test_zero_household_isolated_component_infeasible():
    # 孤立的零户数片区永远无法达标 -> 无解而非悄悄丢弃
    districts = [
        D("d1", (0, 0, 1, 1), 5),
        D("d2", (9, 9, 10, 10), 0),
    ]
    with pytest.raises(MergeInfeasibleError, match="d2"):
        merge_districts(districts, 1)


def test_random_partitions_are_valid_and_conservative():
    """随机链状片区上，方案必须全覆盖、组内连通、组组达标、总户数守恒、组数最大。"""
    rng = random.Random(20261116)
    for _ in range(200):
        n = rng.randint(1, 10)
        households = [rng.randint(0, 9) for _ in range(n)]
        threshold = rng.randint(1, 12)
        districts = [
            D(f"d{i}", (i, 0, i + 1, 1), households[i]) for i in range(n)
        ]
        total = sum(households)
        try:
            groups = merge_districts(districts, threshold)
        except MergeInfeasibleError:
            # 无解时：至少存在一个连通分量（此处全图为一条链即整图）总量低于门槛
            assert total < threshold
            continue

        # 全覆盖且无重复
        flat = [x for g in groups for x in g]
        assert sorted(flat) == sorted(d.id for d in districts)
        hh = {d.id: d.households for d in districts}
        # 守恒 + 达标 + 组内连通（链上同组编号必须连续）
        assert sum(hh[x] for x in flat) == total
        for g in groups:
            assert sum(hh[x] for x in g) >= threshold
            idxs = sorted(int(x[1:]) - 1 for x in g)
            assert idxs == list(range(idxs[0], idxs[-1] + 1))
        # 组数最大：对链做一次独立的线性 DP 校验
        best = [-10**9] * (n + 1)
        best[0] = 0
        prefix = [0]
        for h in households:
            prefix.append(prefix[-1] + h)
        for r in range(1, n + 1):
            for l in range(r):
                if prefix[r] - prefix[l] >= threshold:
                    best[r] = max(best[r], best[l] + 1)
        assert len(groups) == best[n]
