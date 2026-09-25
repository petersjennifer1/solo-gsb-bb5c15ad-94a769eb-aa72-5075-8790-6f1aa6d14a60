"""拼版纸张直线裁切（guillotine）工序单。

服务端按当前拼版参数重新计算指定页，以该页**标签实际矩形**与（第一页的）
**占用区**为禁切区域，递归搜索直线裁切顺序：

- 每条刀线横贯当前纸块（从纸块一边直到对边），不得穿过任何禁切区域，
  并把纸块分成两块；
- 递归直到每个纸块至多包含一个禁切区域（标签或占用区）为止；
- 同一纸块存在多条候选刀线时按**固定顺序**尝试：先竖刀（位置 x 从小到
  大）、再横刀（位置 y 从小到大）；候选位置取自禁切区域的边缘——任何
  可行刀的分块效果都与某条边缘刀相同，因此边缘候选是完备的；
- 失败状态按「禁切区域集合」记忆化缓存：对同一集合而言，有效刀线必然
  落在区域跨度之内，与容纳它的纸块无关，因此同一集合只需展开一次；
  这也让同一参数下输出的步骤顺序完全可复现；
- 整页无法直线裁切时（根集合被判定不可分）不返回任何部分步骤，
  以 feasible=false 与中文说明告知。
"""

from __future__ import annotations

from .layout import EPS, compute_layout
from .models import (
    CutBlock,
    CutStep,
    CuttingResponse,
    LayoutRequest,
)

# 展开状态上限：防止极端交错的禁切区域引发指数级搜索（正常版面远低于此）
_MAX_STATES = 100_000


def _r2(v: float) -> float:
    """输出坐标统一保留 2 位小数（与排版输出一致）。"""
    return round(v + 0.0, 2)


def build_cutting_plan(params: LayoutRequest, page: int) -> CuttingResponse:
    """按当前参数重算拼版，为指定页生成直线裁切工序单。

    页码越界、参数非法或搜索状态超限时抛出中文 ValueError（由接口层转为
    HTTP 400）；无法直线裁切不是错误，以 feasible=false 正常返回。
    """
    plan = compute_layout(params)
    if page < 1 or page > len(plan.pages):
        raise ValueError(
            f"页码超出范围：本次拼版共 {len(plan.pages)} 页，请求第 {page} 页。"
        )
    pg = plan.pages[page - 1]

    # 禁切区域：本页标签的实际矩形；第一页再加上首页占用区
    regions: list[tuple[float, float, float, float]] = [
        (lab.x, lab.y, lab.x + lab.width, lab.y + lab.height)
        for lab in pg.labels
    ]
    label_count = len(regions)
    occupied_count = 0
    if page == 1:
        for z in plan.occupied_zones:
            regions.append((z.x, z.y, z.x + z.width, z.y + z.height))
        occupied_count = len(plan.occupied_zones)

    total = len(regions)
    if total <= 1:
        # 禁切区域不超过 1 个：整纸即为终块，无需下刀
        return CuttingResponse(
            page=page,
            feasible=True,
            label_count=label_count,
            occupied_count=occupied_count,
            steps=[],
            message=None,
        )

    failed: set[frozenset[int]] = set()  # 已判定不可分的区域集合（失败缓存）
    steps: list[CutStep] = []
    states = 0  # 已展开状态数（预算保护）

    def expand(
        block: tuple[float, float, float, float], ids: frozenset[int]
    ) -> bool:
        """尝试把 block 中的区域集合 ids 分离到每块至多一个；成功返回 True。"""
        nonlocal states
        if len(ids) <= 1:
            return True
        if ids in failed:
            return False
        states += 1
        if states > _MAX_STATES:
            raise ValueError(
                "本页裁切规划计算量超出上限：禁切区域交错过于复杂，"
                "请减少本页标签数量或增大裁切间距后重试。"
            )
        bx0, by0, bx1, by1 = block

        # 候选刀线按固定顺序生成：先竖刀（x 升序），再横刀（y 升序）。
        # 位置取自区域边缘，且必须严格位于纸块内部、不穿过任何禁切区域；
        # 两侧都至少含一个区域才是有效分块。
        candidates: list[
            tuple[str, float, frozenset[int], frozenset[int]]
        ] = []
        for c in sorted(
            {regions[i][0] for i in ids} | {regions[i][2] for i in ids}
        ):
            if c <= bx0 + EPS or c >= bx1 - EPS:
                continue
            if any(regions[i][0] + EPS < c < regions[i][2] - EPS for i in ids):
                continue  # 穿过某个禁切区域
            left = frozenset(i for i in ids if regions[i][2] <= c + EPS)
            right = ids - left
            if left and right:
                candidates.append(("vertical", c, left, right))
        for c in sorted(
            {regions[i][1] for i in ids} | {regions[i][3] for i in ids}
        ):
            if c <= by0 + EPS or c >= by1 - EPS:
                continue
            if any(regions[i][1] + EPS < c < regions[i][3] - EPS for i in ids):
                continue
            top = frozenset(i for i in ids if regions[i][3] <= c + EPS)
            bottom = ids - top
            if top and bottom:
                candidates.append(("horizontal", c, top, bottom))

        for direction, c, a_ids, b_ids in candidates:
            if direction == "vertical":
                a_block = (bx0, by0, c, by1)
                b_block = (c, by0, bx1, by1)
                line = (c, by0, c, by1)
            else:
                a_block = (bx0, by0, bx1, c)
                b_block = (bx0, c, bx1, by1)
                line = (bx0, c, bx1, c)
            steps.append(
                CutStep(
                    step=0,  # 整体成功后统一编号
                    direction=direction,
                    position=_r2(c),
                    x1=_r2(line[0]),
                    y1=_r2(line[1]),
                    x2=_r2(line[2]),
                    y2=_r2(line[3]),
                    block=CutBlock(
                        x=_r2(bx0),
                        y=_r2(by0),
                        width=_r2(bx1 - bx0),
                        height=_r2(by1 - by0),
                    ),
                )
            )
            if expand(a_block, a_ids) and expand(b_block, b_ids):
                return True
            steps.pop()  # 该刀线导致不可分，回溯尝试下一候选
        failed.add(ids)
        return False

    root = (0.0, 0.0, plan.page_width, plan.page_height)
    if not expand(root, frozenset(range(total))):
        # 无可行方案：说明该页无法直线裁切，不返回任何部分步骤
        return CuttingResponse(
            page=page,
            feasible=False,
            label_count=label_count,
            occupied_count=occupied_count,
            steps=[],
            message=(
                f"第 {page} 页无法直线裁切：本页 {total} 个禁切区域"
                f"（标签 {label_count} 个、占用区 {occupied_count} 个）相互交错，"
                "任何横贯纸块的直线刀都会穿过禁切区域，无法把每个标签 / 占用区"
                "分离到独立纸块。可增大裁切间距或调整排版参数后重试。"
            ),
        )
    # 先序记录即物理下刀顺序（先切外层纸块，再切分出来的子纸块）
    for k, s in enumerate(steps, 1):
        s.step = k
    return CuttingResponse(
        page=page,
        feasible=True,
        label_count=label_count,
        occupied_count=occupied_count,
        steps=steps,
        message=None,
    )
