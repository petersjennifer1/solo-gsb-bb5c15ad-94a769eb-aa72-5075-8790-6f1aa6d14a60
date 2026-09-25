"""拼版纸张直线裁切（guillotine）工序单。

服务端按当前拼版参数**重新计算完整拼版**，取指定页后，把该页每个
已排入标签的**实际矩形**（含旋转后的宽高）以及（第一页的）首页
**占用区矩形**作为禁切区域（刀线不得穿过的实心矩形），递归搜索
一套全宽 / 全高的直线裁切工序：

- 每一刀是一条横贯当前纸块的直线（竖刀从上边到下边，横刀从左边到
  右边），不得穿过任何禁切区域，并把当前纸块切成两块；
- 递归直至每块至多包含一个标签或占用区（空块也视为终止块，不会再
  产生刀线）；
- 候选刀线只取禁切矩形的边界：竖刀 x = 矩形左 / 右缘，横刀
  y = 矩形上 / 下缘，并固定尝试顺序——**先竖刀、后横刀；同向按坐标
  从小到大**——因此同一输入的工序顺序完全可复现；
- 失败纸块的完整状态（几何边界 + 所含区域集合）在本次搜索内缓存
  （记忆化），失败分支不会被重复搜索；
- 找不到任何完整方案时返回 ``feasible=False`` 且**不返回任何部分步骤**。

标签之间的裁切间距是空白纸条：刀线沿标签边缘（间距的一侧）切下，
间距留在另一半上，因此刀线永远不需要穿过标签矩形。页边距区域内若
存在占用区，刀线同样沿占用区边缘切分。
"""

from __future__ import annotations

import sys

from .layout import EPS, compute_layout
from .models import (
    CutPiece,
    CutPlanResponse,
    CutStep,
    LayoutRequest,
)

# 输出坐标保留的小数位（与拼版坐标一致，标签坐标本身已 2 位小数）
_DIGITS = 2


def _r(v: float) -> float:
    return round(float(v) + 0.0, _DIGITS)


class _Rect:
    """禁切区域（标签或首页占用区的实际矩形），原纸毫米坐标。"""

    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, x0: float, y0: float, x1: float, y1: float):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1


class _Piece:
    """递归中的纸块：矩形 + 其中包含的禁切区域索引。"""

    __slots__ = ("x0", "y0", "x1", "y1", "items")

    def __init__(self, x0: float, y0: float, x1: float, y1: float,
                 items: frozenset[int]):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.items = items

    def key(self) -> tuple:
        return (_r(self.x0), _r(self.y0), _r(self.x1), _r(self.y1),
                tuple(sorted(self.items)))


def build_cut_plan(params: LayoutRequest, page_no: int) -> CutPlanResponse:
    """按当前参数重算拼版并生成第 page_no 页的直线裁切工序单。

    页码越界抛 ValueError（接口层转为 HTTP 400）；拼版参数本身的
    校验同样在 compute_layout 内完成，错误原样上抛。
    """
    plan = compute_layout(params)
    if page_no < 1 or page_no > len(plan.pages):
        raise ValueError(
            f"页码超出范围：当前拼版共 {len(plan.pages)} 页，请求第 {page_no} 页。"
        )
    page = plan.pages[page_no - 1]

    # 禁切区域：本页已排入标签的实际矩形 + 第一页的首页占用区矩形。
    # 坐标直接取自服务端重算结果（不信任前端），旋转标签用旋转后的宽高。
    rects: list[_Rect] = [
        _Rect(lab.x, lab.y, lab.x + lab.width, lab.y + lab.height)
        for lab in page.labels
    ]
    if page_no == 1:
        rects.extend(
            _Rect(z.x, z.y, z.x + z.width, z.y + z.height)
            for z in plan.occupied_zones
        )

    # 整纸为初始纸块；无禁切区域（空白页）时无需任何刀线
    root = _Piece(0.0, 0.0, plan.page_width, plan.page_height,
                  frozenset(range(len(rects))))
    if len(rects) <= 1:
        return CutPlanResponse(page=page_no, feasible=True, steps=[], message=None)

    failed: set[tuple] = set()  # 记忆化：已证明无解的纸块状态
    # 内部刀线元组：(是否竖刀, 位置, 纸块 x0,y0,x1,y1, 刀线 x1,y1,x2,y2)
    solution: list[tuple] = []
    # 递归深度最坏为 禁切区域数 - 1（每刀只分出一个区域）；小标签一页
    # 可达数百个区域，临时放宽解释器递归上限，搜索完成后恢复
    need = 2 * len(rects) + 200
    old_limit = sys.getrecursionlimit()
    raised = old_limit < need
    if raised:
        sys.setrecursionlimit(need)
    try:
        ok = _solve(root, rects, failed, solution)
    finally:
        if raised:
            sys.setrecursionlimit(old_limit)
    if not ok:
        zone_note = (
            f"、{len(plan.occupied_zones)} 个首页占用区"
            if page_no == 1 and plan.occupied_zones else ""
        )
        return CutPlanResponse(
            page=page_no,
            feasible=False,
            steps=[],
            message=(
                f"第 {page_no} 页无法直线裁切：本页 {len(rects)} 个禁切区域"
                f"（{len(page.labels)} 个标签实际矩形{zone_note}）在水平与"
                "垂直方向互相交错，不存在横贯纸块、不穿过禁切区域、且能把"
                "纸块逐次分成两块直到每块至多剩一个标签 / 占用区的刀线序列。"
                "请调整标签尺寸、裁切间距或允许旋转后重新生成拼版。"
            ),
        )

    # _solve 以「先父后子」的深度优先顺序追加刀线，step 即追加序号
    steps = [
        CutStep(
            step=i + 1,
            direction=("vertical" if vertical else "horizontal"),
            x1=_r(x1), y1=_r(y1), x2=_r(x2), y2=_r(y2),
            piece=CutPiece(
                x=_r(px0), y=_r(py0),
                width=_r(px1 - px0), height=_r(py1 - py0),
            ),
        )
        for i, (vertical, at, px0, py0, px1, py1, x1, y1, x2, y2)
        in enumerate(solution)
    ]
    return CutPlanResponse(page=page_no, feasible=True, steps=steps, message=None)


def _contains(rect: _Rect, piece: _Piece) -> bool:
    """矩形（含 EPS 容差）是否整体位于纸块内。"""
    return (
        rect.x0 >= piece.x0 - EPS and rect.y0 >= piece.y0 - EPS
        and rect.x1 <= piece.x1 + EPS and rect.y1 <= piece.y1 + EPS
    )


def _solve(piece: _Piece, rects: list[_Rect], failed: set[tuple],
           out: list) -> bool:
    """在纸块 piece 上递归搜索裁切工序；成功时把刀线追加到 out。

    每个可行纸块至少含 2 个禁切区域（调用方保证）。返回是否找到完整
    方案；失败纸块的完整状态（几何边界 + 所含区域集合）记入 failed：
    竖 / 横刀先后顺序不同可能切出同一个剩余子问题，记忆化后该子问题
    只搜索一次。几何坐标按 2 位小数归一，避免浮点噪声击穿缓存。
    """
    state_key = piece.key()
    if state_key in failed:
        return False

    contained = [
        rects[i] for i in sorted(piece.items) if _contains(rects[i], piece)
    ]
    if len(contained) <= 1:
        return True  # 至多一个禁切区域：终止块，无需下刀
    if len(contained) < len(piece.items):
        # 理论上不会发生（子块由刀线切出，区域整体落在一侧）；防御性处理
        failed.add(state_key)
        return False

    for vertical, at, left, right in _candidate_cuts(piece, rects):
        steps_before = len(out)
        if vertical:
            step = (True, at,
                    piece.x0, piece.y0, piece.x1, piece.y1,
                    at, piece.y0, at, piece.y1)
        else:
            step = (False, at,
                    piece.x0, piece.y0, piece.x1, piece.y1,
                    piece.x0, at, piece.x1, at)
        out.append(step)
        if _solve(left, rects, failed, out) and _solve(right, rects, failed, out):
            return True
        # 本刀无法完成全部子块：回退本刀及其子树追加的步骤，尝试下一候选
        del out[steps_before:]

    failed.add(state_key)
    return False


def _candidate_cuts(
    piece: _Piece, rects: list[_Rect]
) -> list[tuple[bool, float, _Piece, _Piece]]:
    """生成纸块上按固定顺序去重后的候选刀线。

    顺序固定为：先竖刀（x 从小到大），再横刀（y 从小到大），保证工序
    可复现。每条候选刀线必须：

    1. 落在纸块内部（严格离开边界，避免零宽度子块）；
    2. 不穿过任何禁切矩形内部（贴着矩形边缘允许）；
    3. 把禁切区域分成**非空的两组**（两侧都至少有一个区域），即真正
       把纸块分成两块。
    """
    cuts: list[tuple[bool, float, _Piece, _Piece]] = []

    def _collect(vertical: bool) -> list[float]:
        coords: set[float] = set()
        for i in piece.items:
            r = rects[i]
            coords.add(r.x0 if vertical else r.y0)
            coords.add(r.x1 if vertical else r.y1)
        lo = piece.x0 if vertical else piece.y0
        hi = piece.x1 if vertical else piece.y1
        vals = [
            c for c in coords
            if c > lo + EPS and c < hi - EPS
        ]
        return sorted(set(_r(c) for c in vals))

    def _make(vertical: bool, at: float) -> tuple[_Piece, _Piece] | None:
        left_items: set[int] = set()
        right_items: set[int] = set()
        blocked = False
        for i in piece.items:
            r = rects[i]
            if vertical:
                if r.x1 <= at + EPS:
                    left_items.add(i)
                elif r.x0 >= at - EPS:
                    right_items.add(i)
                else:
                    blocked = True  # 刀线穿过矩形内部
                    break
            else:
                if r.y1 <= at + EPS:
                    left_items.add(i)
                elif r.y0 >= at - EPS:
                    right_items.add(i)
                else:
                    blocked = True
                    break
        if blocked or not left_items or not right_items:
            return None
        if vertical:
            a = _Piece(piece.x0, piece.y0, at, piece.y1, frozenset(left_items))
            b = _Piece(at, piece.y0, piece.x1, piece.y1, frozenset(right_items))
        else:
            a = _Piece(piece.x0, piece.y0, piece.x1, at, frozenset(left_items))
            b = _Piece(piece.x0, at, piece.x1, piece.y1, frozenset(right_items))
        return a, b

    for vertical in (True, False):
        for at in _collect(vertical):
            made = _make(vertical, at)
            if made is not None:
                cuts.append((vertical, at, made[0], made[1]))
    return cuts
