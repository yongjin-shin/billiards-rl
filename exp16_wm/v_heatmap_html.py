"""
exp16_wm/v_heatmap_html.py — the value maps of exp16_wm.v_heatmap as one self-contained, rotatable 3-D page (HTML, no network needed).

Reads the cache `v_heatmap` wrote (truth / 1 cm average / V over the cue ball's position for one table layout) and writes a page with
the three surfaces side by side (drag any of them: all turn together), a height scale, a view preset row, a hover read-out and a table
of the summary numbers.

Usage:
    python -m exp16_wm.v_heatmap_html --cache logs/teacher_data/v6/heatmap_left5.npz --out logs/teacher_data/v6/heatmap_3d.html
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TITLE = "수구 위치 값 지도"
NAMES = ("truth", "smooth", "v")
LABELS = {"truth": "① 실제 결과", "smooth": "② 1 cm 평균", "v": "③ V의 예측"}
NOTES = {"truth": "위치마다 시뮬레이터로 한 번 친 값", "smooth": "수구를 1 cm씩 흔들어 8번 친 평균", "v": "학습한 V가 내놓는 값"}


def neighbour_jump_share(z: np.ndarray) -> float:
    """Share of adjacent grid cells (both finite) whose values differ by at least 1 ball."""
    d = np.concatenate([np.abs(z[:, 1:] - z[:, :-1]).ravel(), np.abs(z[1:, :] - z[:-1, :]).ravel()])
    d = d[np.isfinite(d)]
    return float((d >= 1.0).mean()) if len(d) else float("nan")


def panel_stats(z: np.ndarray, xs: np.ndarray, ys: np.ndarray, balls: np.ndarray, w: float, l: float) -> dict[str, float]:
    gx, gy = np.meshgrid(xs, ys)
    ok = np.isfinite(z)
    band = ok & (gx > 0.55 * w) & (gy > 0.85 * l)
    near = np.zeros_like(ok)
    for b in balls:
        near |= np.hypot(gx - b[0], gy - b[1]) < 0.12
    near &= ok
    out = {"mean": float(z[ok].mean()) if ok.any() else float("nan"), "std": float(z[ok].std()) if ok.any() else float("nan"),
           "jump": neighbour_jump_share(z), "band": float(z[band].mean()) if band.any() else float("nan"),
           "near": float(z[near].mean()) if near.any() else float("nan")}
    return {k: (v if np.isfinite(v) else None) for k, v in out.items()}          # JSON has no NaN: an empty region is null (shown as –)


def _list(z: np.ndarray, digits: int) -> list[float | None]:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in z.ravel()]


def page_data(cache: str) -> dict:
    from exp16_wm.networks import TABLE_L, TABLE_W
    from exp16_wm.v_heatmap import live_balls
    c = np.load(cache, allow_pickle=True)
    xs, ys, left = c["xs"], c["ys"], int(c["left"])
    balls = live_balls(c["raw"], TABLE_W, TABLE_L)
    maps = {n: c[n] for n in NAMES}
    return {"W": TABLE_W, "L": TABLE_L, "nx": len(xs), "ny": len(ys), "zmax": left, "xs": [round(float(v), 5) for v in xs],
            "ys": [round(float(v), 5) for v in ys], "balls": [[round(float(b[0]), 4), round(float(b[1]), 4)] for b in balls],
            "ballR": 0.028575, "panels": [{"key": n, "label": LABELS[n], "note": NOTES[n], "z": _list(maps[n], 0 if n == "truth" else 2),
                                           "stats": panel_stats(maps[n], xs, ys, balls, TABLE_W, TABLE_L)} for n in NAMES]}


def explain_figure(png: str) -> str:
    """The <section> that shows the explanation picture (a PNG embedded as a data URI); empty when no picture is given."""
    if not png:
        return ""
    import base64
    with open(png, "rb") as f:
        uri = base64.b64encode(f.read()).decode("ascii")
    return ('<section class="explain"><h2>먼저: "1 cm 흔든다"는 게 무슨 뜻인가</h2>'
            '<figure><img src="data:image/png;base64,' + uri + '" alt="왼쪽: 수구 한 자리를 1 cm 원 안에서 8번 옮겨 쳐 본 결과. 오른쪽: 같은 줄을 따라간 ① 실제 결과, ② 1 cm 평균, ③ V의 예측.">'
            '<figcaption>왼쪽은 한 자리를 1 cm 흔들어 8번 쳐 본 실제 결과입니다. 오른쪽에서 ①은 칸마다 한 번 친 값, ②는 칸마다 이 8번의 평균입니다.</figcaption></figure></section>')


def build_html(cache: str, explain_png: str = "") -> str:
    data = page_data(cache)
    return (TEMPLATE.replace("__TITLE__", TITLE).replace("__EXPLAIN__", explain_figure(explain_png))
            .replace("__DATA__", json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True); p.add_argument("--out", required=True)
    p.add_argument("--explain-png", default="", help="a picture (exp16_wm.v_heatmap_explain) to show above the maps")
    args = p.parse_args()
    html = build_html(args.cache, args.explain_png)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"saved {args.out} ({len(html) / 1024:.0f} KB)", flush=True)


TEMPLATE = r"""<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600&display=swap">
<style>
/* Layout: a header with the question, one control row, three linked 3-D surfaces side by side (stacked on a phone), a legend, a table view. */
:root {
  --bg: #fcfcfb; --fg: #0b0b0b; --muted: #52514e; --line: #dcdbd6; --card: #f4f4f1; --accent: #2a78d6; --floor: #b4b3ac;
  --ball: #fcfcfb; --ring: #0b0b0b;
  --r0: #cde2fb; --r1: #9ec5f4; --r2: #6da7ec; --r3: #3987e5; --r4: #256abf; --r5: #184f95; --r6: #0d366b;
  --font: "Noto Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", system-ui, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #1a1a19; --fg: #f2f1ec; --muted: #c3c2b7; --line: #3a3a37; --card: #232321; --accent: #3987e5; --floor: #5c5b56;
    --ball: #1a1a19; --ring: #f2f1ec;
    --r0: #0d366b; --r1: #184f95; --r2: #256abf; --r3: #3987e5; --r4: #6da7ec; --r5: #9ec5f4; --r6: #cde2fb;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #1a1a19; --fg: #f2f1ec; --muted: #c3c2b7; --line: #3a3a37; --card: #232321; --accent: #3987e5; --floor: #5c5b56;
  --ball: #1a1a19; --ring: #f2f1ec;
  --r0: #0d366b; --r1: #184f95; --r2: #256abf; --r3: #3987e5; --r4: #6da7ec; --r5: #9ec5f4; --r6: #cde2fb;
  color-scheme: dark;
}
body { background: var(--bg); color: var(--fg); font-family: var(--font); line-height: 1.55; padding-inline: 1rem; padding-block: 1.25rem 2rem; }
main { max-width: 78rem; margin-inline: auto; display: flex; flex-direction: column; gap: 1.1rem; }
h1 { font-size: 1.35rem; font-weight: 600; margin: 0; text-wrap: balance; }
.lede { margin: 0.25rem 0 0; color: var(--muted); max-width: 46rem; }
.controls { display: flex; flex-wrap: wrap; gap: 0.6rem 1.2rem; align-items: center; }
.controls .group { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
button { font: inherit; color: var(--fg); background: var(--card); border: 1px solid var(--line); border-radius: 0.4rem; padding: 0.3rem 0.7rem; cursor: pointer; }
button:hover { border-color: var(--accent); }
button:focus-visible, canvas:focus-visible, input:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
label { color: var(--muted); display: inline-flex; gap: 0.4rem; align-items: center; }
input[type=range] { accent-color: var(--accent); width: 9rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 22rem), 1fr)); gap: 1rem; }
.panel { min-width: 0; display: flex; flex-direction: column; gap: 0.35rem; }
.panel h2 { font-size: 1rem; font-weight: 600; margin: 0; }
.panel .note { color: var(--muted); font-size: 0.875rem; margin: 0; }
.stage { position: relative; background: var(--card); border: 1px solid var(--line); border-radius: 0.5rem; }
canvas { display: block; width: 100%; aspect-ratio: 5 / 6; touch-action: none; cursor: grab; border-radius: 0.5rem; }
canvas:active { cursor: grabbing; }
.facts { margin: 0; display: grid; grid-template-columns: 1fr auto; gap: 0.1rem 0.8rem; font-size: 0.875rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.facts dd { margin: 0; color: var(--fg); text-align: right; }
.tip { position: absolute; pointer-events: none; z-index: 2; background: var(--bg); color: var(--fg); border: 1px solid var(--line); border-radius: 0.35rem;
  padding: 0.2rem 0.5rem; font-size: 0.8125rem; font-variant-numeric: tabular-nums; white-space: nowrap; }
.legend { display: flex; align-items: center; gap: 0.6rem; color: var(--muted); font-size: 0.875rem; font-variant-numeric: tabular-nums; }
.legend .bar { flex: 0 1 16rem; height: 0.7rem; border-radius: 0.2rem; border: 1px solid var(--line);
  background: linear-gradient(to right, var(--r0), var(--r1), var(--r2), var(--r3), var(--r4), var(--r5), var(--r6)); }
details { color: var(--muted); }
summary { cursor: pointer; }
.scroll { overflow-x: auto; margin-top: 0.5rem; }
table { border-collapse: collapse; font-size: 0.875rem; font-variant-numeric: tabular-nums; min-width: 28rem; }
th, td { padding: 0.3rem 0.8rem; border-bottom: 1px solid var(--line); text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--fg); font-weight: 600; }
td { color: var(--fg); }
.explain { display: flex; flex-direction: column; gap: 0.5rem; }
.explain h2 { font-size: 1.05rem; font-weight: 600; margin: 0; }
.explain figure { margin: 0; display: flex; flex-direction: column; gap: 0.4rem; }
.explain img { display: block; max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 0.5rem; background: #fcfcfb; }
.explain figcaption { color: var(--muted); font-size: 0.875rem; max-width: 52rem; }
.foot { color: var(--muted); font-size: 0.875rem; max-width: 52rem; margin: 0; }
</style>

<main>
  <header>
    <h1>__TITLE__</h1>
    <p class="lede">공 5개의 배치를 고정하고 수구를 놓는 자리를 바꿔 가며 값을 쟀습니다. 바닥은 수구 위치, 높이와 색은 그 자리에서 놓칠 때까지 연속으로 넣는 공 수입니다. 아무 곳이나 끌면 세 지도가 함께 돌아갑니다.</p>
  </header>

  __EXPLAIN__

  <div class="controls">
    <div class="group" role="group" aria-label="보는 각도">
      <button type="button" id="view-default">기본 각도</button>
      <button type="button" id="view-top">위에서</button>
      <button type="button" id="view-side">옆에서</button>
    </div>
    <label for="zscale">높이 <input type="range" id="zscale" min="0.4" max="2.4" step="0.1" value="1"></label>
    <label for="showballs"><input type="checkbox" id="showballs" checked> 공 위치 표시</label>
  </div>

  <div class="grid" id="grid"></div>

  <div class="legend" aria-hidden="true"><span>0개</span><div class="bar"></div><span id="zmax-label"></span></div>

  <details>
    <summary>표로 보기</summary>
    <div class="scroll"><table id="table"></table></div>
  </details>

  <p class="foot">한 가지 공 배치(남은 공 5개)만 그린 것이라 다른 배치로 일반화하지 않습니다. 칸 사이는 선으로 이은 것이고 칸 간격은 2 cm입니다. 공 근처에는 수구를 놓을 수 없어 구멍으로 비어 있습니다.</p>
</main>

<script>
const D = __DATA__;
const W = D.W, L = D.L, nx = D.nx, ny = D.ny, ZMAX = D.zmax, STEP = D.xs[1] - D.xs[0];
const state = { yaw: 0.6, pitch: 0.65, zs: 1, balls: true };
const PRESETS = { "view-default": [0.6, 0.65], "view-top": [0, 1.5], "view-side": [0.6, 0.18] };
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt = (v, d, k = 1) => (v === null || v === undefined) ? "–" : (v * k).toFixed(d);

let theme = null;
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function rgb(hex) { const h = hex.replace("#", ""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
function loadTheme() {
  const key = (document.documentElement.getAttribute("data-theme") || "") + "|" + matchMedia("(prefers-color-scheme: dark)").matches;
  if (theme && theme.key === key) return theme;
  theme = { key, ramp: [0, 1, 2, 3, 4, 5, 6].map(i => rgb(cssVar("--r" + i))), card: cssVar("--card"), floor: cssVar("--floor"),
            ball: cssVar("--ball"), ring: cssVar("--ring"), fg: cssVar("--fg") };
  return theme;
}
let ramp = null;                                    // the current theme's colour stops, set once per frame
function rampAt(v) {
  const r = ramp, t = clamp(v / ZMAX, 0, 1) * 6, i = Math.min(5, Math.floor(t)), f = t - i;
  return [0, 1, 2].map(c => r[i][c] + (r[i + 1][c] - r[i][c]) * f);
}

const grid = document.getElementById("grid");
const panels = D.panels.map(p => {
  const sec = document.createElement("section"); sec.className = "panel";
  const f = p.stats;
  sec.innerHTML = '<h2>' + p.label + '</h2><p class="note">' + p.note + '</p>' +
    '<div class="stage"><canvas tabindex="0" role="img" aria-label="' + p.label + ' 3차원 지도. 끌어서 돌리고 방향키로도 돌릴 수 있습니다."></canvas><div class="tip" hidden></div></div>' +
    '<dl class="facts"><dt>이웃한 칸이 1개 이상 다른 비율</dt><dd>' + fmt(f.jump, 1, 100) + '%</dd>' +
    '<dt>값의 표준편차</dt><dd>' + fmt(f.std, 2) + '</dd></dl>';
  grid.appendChild(sec);
  const canvas = sec.querySelector("canvas");
  const z = Float32Array.from(p.z, v => v === null ? NaN : v);
  return { p, canvas, ctx: canvas.getContext("2d"), tip: sec.querySelector(".tip"), z, sx: new Float32Array(nx * ny), sy: new Float32Array(nx * ny),
           dp: new Float32Array(nx * ny), quads: new Int32Array((nx - 1) * (ny - 1)), qd: new Float32Array(nx * ny), dpr: 1 };
});
document.getElementById("zmax-label").textContent = ZMAX + "개";

function resize(pn) {
  const dpr = window.devicePixelRatio || 1, w = pn.canvas.clientWidth, h = pn.canvas.clientHeight;
  pn.dpr = dpr; pn.canvas.width = Math.round(w * dpr); pn.canvas.height = Math.round(h * dpr);
}

function draw(pn) {
  const th = loadTheme(), ctx = pn.ctx, c = pn.canvas, w = c.clientWidth, h = c.clientHeight;
  ramp = th.ramp;
  ctx.setTransform(pn.dpr, 0, 0, pn.dpr, 0, 0); ctx.fillStyle = th.card; ctx.fillRect(0, 0, w, h);
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), ce = Math.cos(state.pitch), se = Math.sin(state.pitch);
  const H = 1.3 * state.zs, R0 = Math.hypot(W / 2, L / 2, H / 2), sc = 0.47 * Math.min(w, h) / R0;
  const proj = (x, y, z) => {                       // orthographic: x, y on the table (m), z in balls → screen x, y and depth (far = large)
    const X = x - W / 2, Y = y - L / 2, Zc = z / ZMAX * H - H / 2, x1 = X * cy - Y * sy, y1 = X * sy + Y * cy;
    return [w / 2 + x1 * sc, h / 2 - (y1 * se + Zc * ce) * sc, y1 * ce - Zc * se];
  };
  const z = pn.z;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const k = j * nx + i, v = z[k], q = proj(D.xs[i], D.ys[j], Number.isNaN(v) ? 0 : v);
    pn.sx[k] = q[0]; pn.sy[k] = q[1]; pn.dp[k] = q[2];
  }
  const fl = [[0, 0], [W, 0], [W, L], [0, L]].map(a => proj(a[0], a[1], 0));
  ctx.strokeStyle = th.floor; ctx.lineWidth = 1.2; ctx.beginPath();
  fl.forEach((q, n) => n ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])); ctx.closePath(); ctx.stroke();
  if (state.balls) drawBalls(pn, proj, sc, se, false);
  let nq = 0;
  for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const k0 = j * nx + i, k1 = k0 + 1, k2 = k0 + nx, k3 = k2 + 1;
    if (Number.isNaN(z[k0] + z[k1] + z[k2] + z[k3])) continue;
    pn.quads[nq] = k0; pn.qd[k0] = (pn.dp[k0] + pn.dp[k1] + pn.dp[k2] + pn.dp[k3]) / 4; nq++;
  }
  const order = pn.quads.subarray(0, nq); order.sort((a, b) => pn.qd[b] - pn.qd[a]);
  const hz = H / ZMAX / STEP;                        // slope of one ball of value, per metre of table
  for (let n = 0; n < nq; n++) {
    const k0 = order[n], k1 = k0 + 1, k2 = k0 + nx, k3 = k2 + 1;
    const avg = (z[k0] + z[k1] + z[k2] + z[k3]) / 4;
    const gx = (z[k1] - z[k0] + z[k3] - z[k2]) / 2 * hz, gy = (z[k2] - z[k0] + z[k3] - z[k1]) / 2 * hz;
    const shade = 0.6 + 0.4 / Math.sqrt(1 + gx * gx + gy * gy), col = rampAt(avg).map(v => Math.round(v * shade));
    ctx.fillStyle = ctx.strokeStyle = "rgb(" + col[0] + "," + col[1] + "," + col[2] + ")"; ctx.lineWidth = 0.7;
    ctx.beginPath(); ctx.moveTo(pn.sx[k0], pn.sy[k0]); ctx.lineTo(pn.sx[k1], pn.sy[k1]); ctx.lineTo(pn.sx[k3], pn.sy[k3]); ctx.lineTo(pn.sx[k2], pn.sy[k2]);
    ctx.closePath(); ctx.fill(); ctx.stroke();
  }
  if (state.balls) drawBalls(pn, proj, sc, se, true);
}

function drawBalls(pn, proj, sc, se, top) {       // beneath the surface they sit on the floor; on top they are drawn as rings so they stay visible
  const th = loadTheme(), ctx = pn.ctx;
  D.balls.forEach(b => {
    const q = proj(b[0], b[1], 0), rx = D.ballR * sc, ry = Math.max(1.5, D.ballR * sc * se);
    ctx.beginPath(); ctx.ellipse(q[0], q[1], rx, ry, 0, 0, 2 * Math.PI);
    if (!top) { ctx.fillStyle = th.ball; ctx.fill(); }
    ctx.lineWidth = top ? 1.6 : 1.2; ctx.strokeStyle = th.ring; ctx.globalAlpha = top ? 0.8 : 1; ctx.stroke(); ctx.globalAlpha = 1;
  });
}

let queued = false;
function redraw() { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; panels.forEach(draw); }); }

let drag = null;
panels.forEach(pn => {
  const c = pn.canvas;
  c.addEventListener("pointerdown", e => { c.setPointerCapture(e.pointerId); drag = { x: e.clientX, y: e.clientY }; pn.tip.hidden = true; });
  c.addEventListener("pointermove", e => {
    if (drag) {
      state.yaw += (e.clientX - drag.x) * 0.008; state.pitch = clamp(state.pitch + (e.clientY - drag.y) * 0.006, 0.12, 1.5);
      drag = { x: e.clientX, y: e.clientY }; redraw();
    } else hover(pn, e);
  });
  const end = () => { drag = null; };
  c.addEventListener("pointerup", end); c.addEventListener("pointercancel", end);
  c.addEventListener("pointerleave", () => { pn.tip.hidden = true; });
  c.addEventListener("keydown", e => {
    const d = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, -0.08], ArrowDown: [0, 0.08] }[e.key];
    if (!d) return; e.preventDefault(); state.yaw += d[0]; state.pitch = clamp(state.pitch + d[1], 0.12, 1.5); redraw();
  });
});

function hover(pn, e) {                              // the vertex under the pointer that faces the viewer
  const r = pn.canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  let best = -1, bd = Infinity;
  for (let k = 0; k < nx * ny; k++) {
    if (Number.isNaN(pn.z[k])) continue;
    const dx = pn.sx[k] - mx, dy = pn.sy[k] - my;
    if (dx * dx + dy * dy < 100 && pn.dp[k] < bd) { bd = pn.dp[k]; best = k; }
  }
  if (best < 0) { pn.tip.hidden = true; return; }
  const i = best % nx, j = (best - i) / nx;
  pn.tip.textContent = "x " + (D.xs[i] * 100).toFixed(0) + " cm, y " + (D.ys[j] * 100).toFixed(0) + " cm: " + pn.z[best].toFixed(2) + "개";
  pn.tip.style.left = Math.min(mx + 12, r.width - 150) + "px"; pn.tip.style.top = Math.max(4, my - 28) + "px"; pn.tip.hidden = false;
}

Object.keys(PRESETS).forEach(id => document.getElementById(id).addEventListener("click", () => {
  [state.yaw, state.pitch] = PRESETS[id]; redraw();
}));
document.getElementById("zscale").addEventListener("input", e => { state.zs = parseFloat(e.target.value); redraw(); });
document.getElementById("showballs").addEventListener("change", e => { state.balls = e.target.checked; redraw(); });

const tbl = document.getElementById("table");
const rows = [["평균 (개)", "mean", 2], ["표준편차", "std", 2], ["이웃한 칸이 1개 이상 다른 비율 (%)", "jump", 1], ["위쪽 오른쪽 띠의 평균 (개)", "band", 2], ["공 근처(12 cm 이내)의 평균 (개)", "near", 2]];
tbl.innerHTML = "<thead><tr><th></th>" + D.panels.map(p => "<th>" + p.label + "</th>").join("") + "</tr></thead><tbody>" +
  rows.map(r => "<tr><td>" + r[0] + "</td>" + D.panels.map(p => "<td>" + (r[1] === "jump" ? fmt(p.stats.jump, 1, 100) : fmt(p.stats[r[1]], r[2])) + "</td>").join("") + "</tr>").join("") + "</tbody>";

new ResizeObserver(() => { panels.forEach(resize); redraw(); }).observe(grid);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
panels.forEach(resize); redraw();
</script>
"""


if __name__ == "__main__":
    main()
