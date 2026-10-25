"""
exp16_wm/v_pipeline_figure.py — one figure of how V8 data is made and how V is trained, on one real layout of a V8 file.

Top row: the layout and its grid of cue positions, one cell magnified (the exact position and one copy on each of the 1, 2, 3, 4, 6 cm rings
with the balls each one pocketed; the cell's label is the mean of the five rings), and the label map of the whole layout.
Bottom row: V gives a value for every cell at once (a model's prediction map), both maps are blurred by the Gaussian convolution used in the loss
(exp16_wm.train_vmap), and the loss is the binary cross-entropy between the two blurred maps.

Usage:
    python -m exp16_wm.v_pipeline_figure --file logs/teacher_data/v8/smoke/layouts_200000.npz --model logs/experiments_v2/v5_A --out fig.png
"""

import argparse
import os
import sys

import numpy as np

STEP_M = 0.07                                     # the grid step of the V8 files (7 cm)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def layout_arrays(z: dict, gid: int) -> dict:
    """One layout of a V8 file as grid arrays: obs rows, ij, labels, exact k, per-ring results, left, grid size."""
    m = z["layout_id"] == gid
    return {"obs": z["obs"][m], "ij": z["cell_ij"][m].astype(int), "label": z["k"][m].astype(float), "exact": z["k_exact"][m].astype(float),
            "ring_k": z["ring_k"][m].astype(int), "radii_mm": z["radii_mm"], "left": int(z["left"][m][0]), "nxy": tuple(int(v) for v in z["grid_nxy"]),
            "xy": z["cell_xy"][m]}


def to_map(values: np.ndarray, ij: np.ndarray, nxy: tuple[int, int]) -> np.ndarray:
    """(nx, ny) map with NaN on cells that are not in ij."""
    g = np.full(nxy, np.nan)
    g[ij[:, 0], ij[:, 1]] = values
    return g


def story_cell(arr: dict) -> int:
    """The cell whose five ring results disagree most (and differ from the exact one), away from the cushions — the clearest example."""
    rk = arr["ring_k"][:, :, 0].astype(float)
    rk = np.where(rk >= 0, rk, np.nan)
    sd = np.nan_to_num(np.nanstd(rk, axis=1))
    inner = (arr["ij"][:, 0] > 1) & (arr["ij"][:, 0] < arr["nxy"][0] - 2) & (arr["ij"][:, 1] > 1) & (arr["ij"][:, 1] < arr["nxy"][1] - 2)
    return int(np.argmax(sd * inner + 0.01 * np.abs(arr["exact"] - np.nan_to_num(np.nanmean(rk, axis=1))) * inner))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True); p.add_argument("--layout", type=int, default=-1, help="layout id (default: the first with 5 balls left)")
    p.add_argument("--model", default="", help="a trained V run dir for the example prediction map"); p.add_argument("--sigma-cm", type=float, default=8.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    z = dict(np.load(args.file, allow_pickle=True))
    ids = np.unique(z["layout_id"])
    lefts = {int(g): int(z["left"][z["layout_id"] == g][0]) for g in ids}
    gid = args.layout if args.layout >= 0 else next((g for g, l in lefts.items() if l == 5), max(lefts, key=lefts.get))
    arr = layout_arrays(z, gid)
    raw = arr["obs"][0]
    pred = None
    if args.model:
        from exp16_wm.v_map_eval import predict_model
        pred = predict_model(args.model, arr["obs"], np.full(len(arr["obs"]), arr["left"]))
    draw(arr, raw, pred, args.sigma_cm / (STEP_M * 100.0), args.out)


def draw(arr: dict, raw: np.ndarray, pred: np.ndarray | None, sigma_cells: float, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle, FancyArrowPatch
    from exp16_wm.networks import TABLE_L, TABLE_W
    from exp16_wm.train_vmap import blur_masked, gaussian_kernel, map_loss, to_grid
    from exp16_wm.v_heatmap import live_balls
    surface, ink, ink2, grid_c = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"
    plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False})
    ramp = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    ramp.set_bad(surface)
    left, nxy, ij, R, step = arr["left"], arr["nxy"], arr["ij"], 0.028575, STEP_M
    balls = live_balls(raw, TABLE_W, TABLE_L)
    pockets = [(0, 0), (TABLE_W, 0), (0, TABLE_L / 2), (TABLE_W, TABLE_L / 2), (0, TABLE_L), (TABLE_W, TABLE_L)]
    ext = (R - step / 2, R + (nxy[0] - 0.5) * step, R - step / 2, R + (nxy[1] - 0.5) * step)
    label_map = to_map(arr["label"], ij, nxy)
    kernel = gaussian_kernel(sigma_cells)
    fig = plt.figure(figsize=(17.5, 10.6), facecolor=surface)
    gs = fig.add_gridspec(2, 4, left=0.02, right=0.99, top=0.88, bottom=0.04, wspace=0.16, hspace=0.30, width_ratios=[1, 1, 1, 1])
    fig.text(0.02, 0.975, "데이터를 어떻게 만들고 V를 어떻게 학습하나 (공 배치 하나를 실제로 따라간 예)", fontsize=16, color=ink, va="top")
    fig.text(0.02, 0.94, "공 배치 하나 = 학습 샘플 1개. 위는 데이터 만들기(서버 시뮬레이션), 아래는 학습(모델 + convolution + 손실).", fontsize=11.5, color=ink2, va="top")

    def table(ax, title, sub=""):
        ax.set_facecolor(surface)
        for b in balls:
            ax.add_patch(Circle(b, R, facecolor=surface, edgecolor=ink, linewidth=1.3, zorder=6))
        for pk in pockets:
            ax.add_patch(Circle(pk, 0.032, facecolor=ink, edgecolor="none", zorder=7))
        ax.set_xlim(-0.04, TABLE_W + 0.04); ax.set_ylim(-0.04, TABLE_L + 0.04); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(ink2); sp.set_linewidth(0.8)
        ax.set_title(title, color=ink, fontsize=12.5, loc="left", pad=22)
        if sub:
            ax.text(0, 1.012, sub, transform=ax.transAxes, color=ink2, fontsize=9.8, va="bottom")

    def show(ax, m, vmin=0, vmax=left):
        return ax.imshow(np.ma.masked_invalid(m.T), origin="lower", extent=ext, cmap=ramp, vmin=vmin, vmax=vmax, interpolation="nearest")

    # 1: layout and the grid of cue positions
    a1 = fig.add_subplot(gs[0, 0]); table(a1, "① 공 배치 s와 수구를 놓을 칸", f"공 {left}개, 7 cm 간격 격자 {nxy[0]}×{nxy[1]}, 공과 겹치는 칸은 뺌: {len(ij)}칸")
    a1.scatter(R + ij[:, 0] * step, R + ij[:, 1] * step, s=9, color="#6da7ec", zorder=3)
    c = story_cell(arr)
    cx, cy = R + ij[c, 0] * step, R + ij[c, 1] * step
    a1.add_patch(Circle((cx, cy), 0.07, fill=False, edgecolor="#eb6834", linewidth=2.0, zorder=8))
    # 2: one cell magnified, six simulations
    a2 = fig.add_subplot(gs[0, 1])
    a2.set_facecolor(surface); a2.set_xlim(-6.8, 6.8); a2.set_ylim(-6.8, 6.8); a2.set_aspect("equal")
    a2.set_title("② 칸 하나에서 6번 시뮬레이션", color=ink, fontsize=12.5, loc="left", pad=22)
    a2.text(0, 1.012, "정확한 위치 1번 + 1·2·3·4·6 cm 거리에서 각 1번 (방향은 무작위, 그림은 예시)", transform=a2.transAxes, color=ink2, fontsize=9.8, va="bottom")
    radii = arr["radii_mm"] / 10.0
    for r in radii:
        a2.add_patch(Circle((0, 0), r, fill=False, edgecolor=grid_c, linestyle=(0, (4, 3)), linewidth=1.2))
    angles = np.deg2rad(np.array([75, 150, 225, 300, 20]))
    ring_vals = arr["ring_k"][c, :, 0].astype(float)
    for r, ang, v in zip(radii, angles, ring_vals):
        x, y = r * np.cos(ang), r * np.sin(ang)
        if v < 0:
            continue
        a2.scatter([x], [y], s=330, color=ramp(v / left), edgecolor=ink, linewidth=1.0, zorder=5)
        a2.text(x, y, f"{int(v)}", ha="center", va="center", color="#ffffff" if v >= 2 else ink, fontsize=10.5, fontweight="bold", zorder=6)
    a2.scatter([0], [0], s=420, marker="X", color=ramp(arr["exact"][c] / left), edgecolor=ink, linewidth=1.5, zorder=7)
    placed = ring_vals[ring_vals >= 0]
    a2.text(0.0, -0.03, f"X: 정확한 위치에서 친 결과 = {int(arr['exact'][c])}개\n● 5곳의 결과 = {', '.join(str(int(v)) for v in placed)} (1·2·3·4·6 cm 순)\n→ 이 칸의 정답 = 5곳의 평균 = {placed.mean():.1f}개",
            transform=a2.transAxes, color=ink, fontsize=10.8, va="top", linespacing=1.55)
    a2.set_xticks([]); a2.set_yticks([])
    for sp in a2.spines.values():
        sp.set_color("#eb6834"); sp.set_linewidth(1.6)
    # 3: label map
    a3 = fig.add_subplot(gs[0, 2]); table(a3, "③ 이 배치의 정답 지도", "모든 칸에 ②를 반복: 칸마다 5곳 평균 = 그 칸의 정답")
    im3 = show(a3, label_map)
    a3.add_patch(Circle((cx, cy), 0.07, fill=False, edgecolor="#eb6834", linewidth=2.0, zorder=8))
    # 4: text
    a4 = fig.add_subplot(gs[0, 3]); a4.axis("off")
    a4.text(0, 1.0, "데이터 한 묶음", fontsize=12.5, color=ink, va="top")
    a4.text(0, 0.93, f"· 배치 하나에서 칸 약 {len(ij)}곳 × 6번 = 약 {len(ij) * 6:,}번 시뮬레이션\n· 서버 한 대(64 vCPU)로 배치 약 1,400개/시간\n· 4시간이면 배치 약 5,600개\n· 정답 지도는 배치마다 따로 한 장\n\n"
            "· 학습 샘플 수 = 공 배치의 수\n· 평가용 배치 100개는 학습에 쓰지 않음", fontsize=11, color=ink2, va="top", linespacing=1.7)
    # 5: model -> prediction map
    a5 = fig.add_subplot(gs[1, 0]); table(a5, "④ 모델이 모든 칸의 값을 한 번에 냄", "입력: 공 배치 + 칸의 수구 위치 → V → 그 칸의 값(공 수)")
    if pred is not None:
        im5 = show(a5, to_map(pred, ij, nxy))
        a5.text(0.0, -0.015, "예: 지금 V(학습된 encoder로 시작)의 예측", transform=a5.transAxes, color=ink2, fontsize=9.8, va="top")
    else:
        a5.text(0.5, 0.5, "(모델 예측 없음)", transform=a5.transAxes, ha="center", color=ink2)
    # 6/7: blurred prediction and blurred label (the loss inputs)
    def blurred(vals):
        g, m = to_grid(torch.as_tensor(np.clip(vals / max(1, left), 0, 1), dtype=torch.float32), torch.as_tensor(ij), nxy)
        return blur_masked(g, m, kernel)[0, 0].numpy() * left, m[0, 0].numpy()
    a6 = fig.add_subplot(gs[1, 1]); table(a6, "⑤ 예측 지도를 convolution(블러 8 cm)", "빈칸은 무시하고 이웃 칸을 가우시안으로 평균")
    a7 = fig.add_subplot(gs[1, 2]); table(a7, "⑥ 정답 지도도 같은 convolution", "같은 블러를 정답에도 걸어서 비교할 대상으로 삼음")
    bl, mask = blurred(arr["label"])
    show(a7, np.where(mask > 0, bl, np.nan))
    loss_txt = ""
    if pred is not None:
        bp, _ = blurred(pred)
        show(a6, np.where(mask > 0, bp, np.nan))
        frac = lambda v: torch.as_tensor(np.clip(v / max(1, left), 1e-6, 1 - 1e-6), dtype=torch.float32)
        lv = float(map_loss(frac(pred), frac(arr["label"]), torch.as_tensor(ij), nxy, kernel))
        loss_txt = f"\n\n이 배치에서 예시 모델의 손실 = {lv:.3f}"
    # 8: loss text
    a8 = fig.add_subplot(gs[1, 3]); a8.axis("off")
    a8.text(0, 1.0, "⑦ 손실과 학습", fontsize=12.5, color=ink, va="top")
    a8.text(0, 0.93, "손실 = 블러한 예측 지도(⑤)와 블러한 정답 지도(⑥)의\n칸별 BCE 평균 (값은 남은 공 수로 나눈 비율)\n\n"
            "· 배치 4개씩 묶어 한 번에 계산 → 역전파 → 갱신\n· 학습 배치 90% / 검증 배치 10%로 나눠 검증 loss가\n  가장 낮은 epoch의 모델을 선택\n· 정책·made encoder는 고정, V 부분만 학습\n\n"
            "블러 0 cm로 하면 칸 하나씩 비교하는 점 단위 학습과 같음" + loss_txt, fontsize=11, color=ink2, va="top", linespacing=1.7)
    # arrows between the stages, from the panel positions
    def arrow(src, dst, kind="right", both=False):
        ps, pd = src.get_position(), dst.get_position()
        if kind == "right":
            x0, y0, x1, y1 = ps.x1 + 0.006, (ps.y0 + ps.y1) / 2, pd.x0 - 0.006, (pd.y0 + pd.y1) / 2
        else:
            x0, y0, x1, y1 = (ps.x0 + ps.x1) / 2, ps.y0 - 0.052, (pd.x0 + pd.x1) / 2, pd.y1 + 0.05
        fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="<|-|>" if both else "-|>", mutation_scale=18, color=ink2, linewidth=1.6))
    arrow(a1, a2); arrow(a2, a3); arrow(a5, a6); arrow(a3, a7, "down"); arrow(a7, a8)
    arrow(a6, a7, both=True)
    pa, pb = a6.get_position(), a7.get_position()
    fig.text((pa.x1 + pb.x0) / 2, (pa.y0 + pa.y1) / 2 + 0.018, "비교", ha="center", fontsize=10, color=ink2)
    cax = fig.add_axes([0.745, 0.58, 0.008, 0.2]); cb = fig.colorbar(im3, cax=cax, ticks=range(0, left + 1)); cb.ax.tick_params(colors=ink2, labelsize=8); cb.ax.set_title("공 수", color=ink2, fontsize=9, pad=4)
    fig.savefig(out, dpi=135, facecolor=surface, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
