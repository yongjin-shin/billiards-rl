"""
v16 inference 과정 시각화
- s_0 하나만 주고 T=60 스텝 rollout
- 매 스텝: z_t → decode(ar_state) → transition → z_{t+1}
- 예측 vs GT 비교
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec

from world_model.ssm_model import SSMWorldModel, LATENT_DIM
from world_model.generate_data_fixeddt import DT
from world_model.wm_predictor import TABLE_W, TABLE_H
from world_model.val_cache import load_val_cache

for fname in ['AppleGothic', 'NanumGothic']:
    try:
        fm.findfont(fm.FontProperties(family=fname), fallback_to_default=False)
        plt.rcParams['font.family'] = fname
        break
    except Exception:
        continue

device = 'cpu'
CKPT  = 'world_model/results/ssm_v16_5cls/best.pt'
T     = 60

TYPE_NAMES  = ['no_coll', 'ball_ball', 'linear', 'circular', 'pocket']
TYPE_COLORS = ['#404060', '#5588ff', '#44cc88', '#cc8844', '#ff5566']

# ── 모델 로드 ──────────────────────────────────────────────
model = SSMWorldModel(LATENT_DIM).to(device)
ckpt  = torch.load(CKPT, map_location=device, weights_only=False)
model.load_state_dict(ckpt['state'])
model.eval()

_val_eps = load_val_cache()

pick = 232  # TP=4, FP=1, recall=0.67, prec=0.80, err=7.3cm
ep_s, ep_f, ep_t, _ = _val_eps[pick]
ep_idx = pick

ep_f = ep_f[:T]
ep_t = ep_t[:T]
ep_s = ep_s[:T+1]

ts = np.arange(T + 1) * DT   # 시간축

# ── inference rollout ──────────────────────────────────────
s0 = torch.from_numpy(ep_s[0:1]).float()  # (1,14)

with torch.no_grad():
    s_hat, type_logit = model(s0, T)

s_hat_np      = s_hat[0].numpy()           # (T+1, 14)
type_logit_np = type_logit[0].numpy()      # (T, 5)
type_probs_np = torch.softmax(type_logit[0], dim=-1).numpy()  # (T, 5)
pred_type     = type_logit_np.argmax(-1)   # (T,)
pred_coll     = pred_type != 0             # (T,) bool

gt = ep_s[:T+1]  # (T+1, 14)

# cm 변환
def to_cm_cue(s):
    x = s[:, 0] * TABLE_W * 100
    y = s[:, 1] * TABLE_H * 100
    return x, y

def to_cm_tgt(s):
    x = s[:, 7] * TABLE_W * 100
    y = s[:, 8] * TABLE_H * 100
    return x, y

# ── Figure ─────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14), facecolor='#0d0f1a')
gs  = GridSpec(4, 3, figure=fig,
               hspace=0.45, wspace=0.32,
               top=0.93, bottom=0.07, left=0.06, right=0.97,
               height_ratios=[1.8, 1.4, 1.0, 1.4])

spine_c = '#2a2d45'

def sax(ax, title='', xl='', yl=''):
    ax.set_facecolor('#0d0f1a')
    ax.tick_params(colors='#6068a0', labelsize=8)
    for sp in ax.spines.values(): sp.set_color(spine_c)
    if title: ax.set_title(title, color='#c8d0f0', fontsize=9.5,
                           fontweight='bold', pad=4)
    if xl: ax.set_xlabel(xl, color='#8890b0', fontsize=8.5)
    if yl: ax.set_ylabel(yl, color='#8890b0', fontsize=8.5)

# ── [0,:] cue & tgt 위치 ──────────────────────────────────
ax_cx = fig.add_subplot(gs[0, :2])
sax(ax_cx, 'cue ball  —  position (cm)  GT vs Pred', 'time (s)', 'cm')
gx, gy   = to_cm_cue(gt)
px, py   = to_cm_cue(s_hat_np)
ax_cx.plot(ts, gx,  color='#5588ff', lw=1.8, label='GT x')
ax_cx.plot(ts, gy,  color='#88aaff', lw=1.8, ls='--', label='GT y')
ax_cx.plot(ts, px,  color='#ff6644', lw=1.4, alpha=0.85, label='Pred x')
ax_cx.plot(ts, py,  color='#ffaa88', lw=1.4, alpha=0.85, ls='--', label='Pred y')
ax_cx.legend(fontsize=8, facecolor='#1a1d2e', edgecolor=spine_c, labelcolor='#c0c8e8', ncol=4)

ax_tx = fig.add_subplot(gs[0, 2])
sax(ax_tx, 'tgt ball  —  position (cm)', 'time (s)', 'cm')
gx2, gy2 = to_cm_tgt(gt)
px2, py2 = to_cm_tgt(s_hat_np)
ax_tx.plot(ts, gx2, color='#5588ff', lw=1.8, label='GT x')
ax_tx.plot(ts, gy2, color='#88aaff', lw=1.8, ls='--', label='GT y')
ax_tx.plot(ts, px2, color='#ff6644', lw=1.4, alpha=0.85, label='Pred x')
ax_tx.plot(ts, py2, color='#ffaa88', lw=1.4, alpha=0.85, ls='--', label='Pred y')
ax_tx.legend(fontsize=7.5, facecolor='#1a1d2e', edgecolor=spine_c, labelcolor='#c0c8e8', ncol=2)

# 충돌 배경
for ax_ in [ax_cx, ax_tx]:
    for t_i in np.where(ep_f)[0]:
        ax_.axvspan(t_i*DT, (t_i+1)*DT, alpha=0.25,
                    color=TYPE_COLORS[ep_t[t_i]], lw=0)

# ── [1,:] type 예측 확률 히트맵 ──────────────────────────
ax_tp = fig.add_subplot(gs[1, :])
sax(ax_tp, 'type_head(z_t)  →  softmax  →  type prob per step',
    'time (s)', 'type')
im = ax_tp.imshow(type_probs_np.T, aspect='auto', origin='lower',
                  extent=[ts[0], ts[T-1], -0.5, 4.5],
                  cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
ax_tp.set_yticks(range(5))
ax_tp.set_yticklabels(TYPE_NAMES, fontsize=8, color='#8890b0')
ax_tp.tick_params(axis='x', colors='#6068a0', labelsize=8)
cbar = fig.colorbar(im, ax=ax_tp, fraction=0.012, pad=0.01)
cbar.ax.tick_params(colors='#8890b0', labelsize=7)

# GT 충돌 마커
for t_i in np.where(ep_f)[0]:
    ax_tp.scatter(t_i*DT, ep_t[t_i], marker='x', s=60,
                  color=TYPE_COLORS[ep_t[t_i]], zorder=5, lw=1.5)

# ── [2,:] GT vs Pred collision 비교 ───────────────────────
ax_coll = fig.add_subplot(gs[2, :])
sax(ax_coll, 'collision detection  —  GT(×) vs Pred(▲)', 'time (s)', '')
ax_coll.set_ylim(-0.5, 1.5)
ax_coll.set_yticks([0, 1])
ax_coll.set_yticklabels(['no coll', 'collision'], fontsize=8.5, color='#8890b0')

# GT
for t_i in np.where(ep_f)[0]:
    ax_coll.scatter(t_i*DT, 1, marker='x', s=80,
                    color=TYPE_COLORS[ep_t[t_i]], zorder=5, lw=2)
# Pred
for t_i in np.where(pred_coll)[0]:
    match = ep_f[t_i]
    ax_coll.scatter(t_i*DT, 0.5, marker='^', s=50,
                    color='#44cc88' if match else '#ff5555',
                    alpha=0.85, zorder=4)

# 범례
from matplotlib.lines import Line2D
leg = [Line2D([0],[0], marker='x', color='#5588ff', lw=0, ms=8, label='GT collision'),
       Line2D([0],[0], marker='^', color='#44cc88', lw=0, ms=7, label='TP (correct)'),
       Line2D([0],[0], marker='^', color='#ff5555', lw=0, ms=7, label='FP (false alarm)')]
ax_coll.legend(handles=leg, fontsize=8, facecolor='#1a1d2e',
               edgecolor=spine_c, labelcolor='#c0c8e8')

# ── [3,:] 위치 오차 (cm) per step ────────────────────────
ax_err = fig.add_subplot(gs[3, :])
sax(ax_err, 'position error per step  (cm)', 'time (s)', 'error (cm)')

cue_err = np.sqrt(((s_hat_np[1:,0]-gt[1:,0])*TABLE_W*100)**2 +
                  ((s_hat_np[1:,1]-gt[1:,1])*TABLE_H*100)**2)
tgt_err = np.sqrt(((s_hat_np[1:,7]-gt[1:,7])*TABLE_W*100)**2 +
                  ((s_hat_np[1:,8]-gt[1:,8])*TABLE_H*100)**2)
mean_err = (cue_err + tgt_err) / 2

ax_err.plot(ts[1:], cue_err, color='#5588ff', lw=1.2, alpha=0.8, label='cue err')
ax_err.plot(ts[1:], tgt_err, color='#ff8844', lw=1.2, alpha=0.8, label='tgt err')
ax_err.plot(ts[1:], mean_err, color='#ffffff', lw=1.8, label=f'mean  (avg={mean_err.mean():.1f}cm)')
ax_err.fill_between(ts[1:], mean_err, alpha=0.1, color='#ffffff')

# 충돌 마커
for t_i in np.where(ep_f)[0]:
    ax_err.axvspan(t_i*DT, (t_i+1)*DT, alpha=0.2,
                   color=TYPE_COLORS[ep_t[t_i]], lw=0)

ax_err.legend(fontsize=8.5, facecolor='#1a1d2e', edgecolor=spine_c,
              labelcolor='#c0c8e8', ncol=3)

# ── 수치 요약 ──────────────────────────────────────────────
gt_c   = ep_f.sum()
tp     = (ep_f & pred_coll).sum()
fp     = (~ep_f & pred_coll).sum()
fn     = (ep_f & ~pred_coll).sum()
recall = tp / gt_c if gt_c > 0 else 0
prec   = tp / pred_coll.sum() if pred_coll.sum() > 0 else 0

info = (f"ep={ep_idx}  |  mean_err={mean_err.mean():.1f}cm  "
        f"|  GT colls={gt_c}  TP={tp}  FP={fp}  FN={fn}  "
        f"|  recall={recall:.3f}  prec={prec:.3f}")
fig.suptitle(f'v16  pure inference  (ss=0.0)  —  {info}',
             color='#c8d0f0', fontsize=11, fontweight='bold')

plt.savefig('world_model/inference_viz.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"Saved → world_model/inference_viz.png")
print(info)
plt.show()
