"""片区合并纯函数模块：共边邻接、连通分量、达标分组最优化。

输入为当前已发布结果中的片区（编号、整数矩形、户数）与统计员设定的最低户数。
规则：
- 两个片区仅当矩形共享一段**正长度的边**时相邻；只在角点相接不算相邻。
- 所有片区先划入共边连通组，分组只能在连通组内部进行（合并不跨越不相邻片区）。
- 每个合并组总户数必须达到最低户数门槛；在“组数最多”的方案中，
  按组内编号升序、组列表升序（字典序最小）确定唯一结果。
- 总户数在合并前后守恒。
- 某连通组整体户数仍低于门槛时无解，调用方据此拒绝发布并说明原因。

本模块不触碰数据库与 Web 层，便于单元测试。
"""
from __future__ import annotations

from dataclasses import dataclass

from .allocation import _id_sort_key

# 矩形表示为 (x1, y1, x2, y2)，均为整数且 x1<x2、y1<y2。
RectTuple = tuple[int, int, int, int]


class MergeInfeasibleError(Exception):
    """不存在可发布的合并方案（例如某连通组总户数低于门槛）。"""


@dataclass(frozen=True)
class MergeDistrict:
    id: str
    rect: RectTuple
    households: int


def share_edge(a: RectTuple, b: RectTuple) -> bool:
    """两个轴对齐整数矩形是否共享一段正长度的边。

    仅角点相接（公共部分退化为一个点）或完全不相交时返回 False；
    同版片区矩形内部不重叠，因此无需考虑重叠情形。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # 竖直公共边：一条的左边/右边与另一条的右边/左边重合，且 y 方向投影重叠长度为正
    if ax2 == bx1 or ax1 == bx2:
        if min(ay2, by2) - max(ay1, by1) > 0:
            return True
    # 水平公共边：上下边重合，且 x 方向投影重叠长度为正
    if ay2 == by1 or ay1 == by2:
        if min(ax2, bx2) - max(ax1, bx1) > 0:
            return True
    return False


def build_adjacency(districts: list[MergeDistrict]) -> list[int]:
    """返回每个片区的邻接位掩码：第 j 位为 1 表示与片区 j 共边。"""
    n = len(districts)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if share_edge(districts[i].rect, districts[j].rect):
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return adj


def connected_components(n: int, adj: list[int]) -> list[tuple[int, ...]]:
    """按邻接位掩码求连通分量，返回各分量节点下标（分量按下标最小者升序）。"""
    seen = [False] * n
    components: list[tuple[int, ...]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: list[int] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            mask = adj[node]
            while mask:
                bit = mask & -mask
                nxt = bit.bit_length() - 1
                mask ^= bit
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        components.append(tuple(sorted(comp)))
    return components


def _component_rank(districts: list[MergeDistrict], nodes: tuple[int, ...]) -> list[int]:
    """分量内节点按编号升序后的名次表。

    rank[i] = 节点 i 在分量内按编号升序排列后的位置，
    于是“组内编号升序、组列表升序”可直接用名次整数的字典序比较实现。
    """
    ordered = sorted(nodes, key=lambda i: _id_sort_key(districts[i].id))
    rank = [0] * len(districts)
    for pos, node in enumerate(ordered):
        rank[node] = pos
    return rank


def _bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _best_partition_for_component(
    districts: list[MergeDistrict],
    nodes: tuple[int, ...],
    adj: list[int],
    min_households: int,
) -> tuple[tuple[int, ...], ...] | None:
    """在单个连通分量内求达标门槛下组数最多、并列时字典序最小的划分。

    位掩码 DP：dp[mask] 为覆盖 mask 的最优划分（组为名次元组，划分按字典序排列），
    不可行为 None。比较键为 (-组数, 划分)：先最大化组数，再取字典序最小。
    分量规模上限为每版 12 个片区，2^12 可直接枚举。
    """
    m = len(nodes)
    rank = _component_rank(districts, nodes)

    # 分量内部的局部邻接掩码与户数
    local_adj = [0] * m
    population = [0] * m
    for i, node in enumerate(nodes):
        population[i] = districts[node].households
        for j, other in enumerate(nodes):
            if adj[node] & (1 << other):
                local_adj[i] |= 1 << j

    size = 1 << m
    connected = [False] * size
    weight = [0] * size
    connected[0] = True
    for mask in range(1, size):
        bit = mask & -mask
        i = bit.bit_length() - 1
        rest = mask ^ bit
        weight[mask] = weight[rest] + population[i]
        # 单点子集连通；多点子集要求 i 与 rest 中至少一个节点相邻且 rest 自身连通
        connected[mask] = rest == 0 or bool(local_adj[i] & rest and connected[rest])

    # dp[mask] 的每组是按编号名次升序的名次元组；整个划分也按字典序排列
    dp: list[tuple[tuple[int, ...], ...] | None] = [None] * size
    dp[0] = ()
    for mask in range(1, size):
        best_key: tuple[int, tuple[tuple[int, ...], ...]] | None = None
        best: tuple[tuple[int, ...], ...] | None = None
        sub = mask
        while sub:
            rest = mask ^ sub
            if (
                connected[sub]
                and weight[sub] >= min_households
                and dp[rest] is not None
            ):
                group = tuple(sorted(rank[nodes[i]] for i in _bits(sub)))
                candidate = tuple(sorted(dp[rest] + (group,)))
                # 组数最多优先，并列取字典序最小
                key = (-len(candidate), candidate)
                if best_key is None or key < best_key:
                    best_key, best = key, candidate
            sub = (sub - 1) & mask
        dp[mask] = best
    return dp[size - 1]


def merge_districts(
    districts: list[MergeDistrict], min_households: int
) -> list[list[str]]:
    """计算合并方案，返回有序的组列表（每组为按编号升序的片区编号列表）。

    组数在所有可行方案中最多；并列时按组内编号升序、组列表升序取字典序最小者。
    无可行方案时抛 MergeInfeasibleError，错误信息说明原因（不发布部分结果）。
    """
    if min_households < 1:
        raise ValueError("最低户数必须为正整数")
    if not districts:
        raise MergeInfeasibleError("当前已发布结果不含任何片区，无法合并")

    adj = build_adjacency(districts)
    components = connected_components(len(districts), adj)

    # 整体可行性：任一连通组总户数低于门槛即无解——
    # 组不能跨越连通分量，欠账分量中的片区无处可去。
    failures: list[str] = []
    for comp in components:
        comp_total = sum(districts[i].households for i in comp)
        if comp_total < min_households:
            ids = "、".join(
                districts[i].id
                for i in sorted(comp, key=lambda i: _id_sort_key(districts[i].id))
            )
            failures.append(
                f"连通组（片区 {ids}）总户数 {comp_total} 低于最低户数 {min_households}"
            )
    if failures:
        raise MergeInfeasibleError(
            "不存在全部片区均达标的合并方案：" + "；".join(failures)
        )

    result_groups: list[tuple[str, ...]] = []
    for comp in components:
        partition = _best_partition_for_component(districts, comp, adj, min_households)
        if partition is None:
            # 理论上被前置可行性检查覆盖，保留防御性分支
            raise MergeInfeasibleError("合并方案求解失败")
        ordered_nodes = sorted(comp, key=lambda i: _id_sort_key(districts[i].id))
        for rank_tuple in partition:
            result_groups.append(
                tuple(districts[ordered_nodes[r]].id for r in rank_tuple)
            )

    # 分量之间按组的编号字典序统一排序，保证全局“组列表升序”确定
    result_groups.sort()

    # 守恒护栏：合并不改变总户数，且每个片区恰好出现一次
    total_before = sum(d.households for d in districts)
    households_by_id = {d.id: d.households for d in districts}
    total_after = 0
    covered: set[str] = set()
    output: list[list[str]] = []
    for group in result_groups:
        output.append(list(group))
        for district_id in group:
            if district_id in covered:
                raise MergeInfeasibleError("合并方案存在重复片区，拒绝发布")
            covered.add(district_id)
            total_after += households_by_id[district_id]
    if covered != set(households_by_id) or total_after != total_before:
        raise MergeInfeasibleError("合并方案未覆盖全部片区或总户数不守恒，拒绝发布")
    return output
