"""
GT 데이터 시간축 시각화
- 실제 에피소드 1개의 state/collision/gt_ar를 시간 순서로 표시
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm

for fname in ['AppleGothic', 'NanumGothic', 'Malgun Gothic']:
    try:
        fm.findfont(fm.FontProperties(family=fname), fallback_to_default=False)
        plt.rcParams['font.family'] = fname
        break
    except Exception:
        continue

DT = 0.05
TABLE_W, TABLE_H = 1.116, 0.635   # m (normalized 기준)

# ── 데이터 로드: ball_ball 충돌 포함 에피소드 선택 ──
d = np.load("world_model/data_fixeddt/sac_sac5m_20260905_210839.npz")
states     = d["states"]       # (N, T_MAX, 14)
coll_flags = d["coll_flags"]   # (N, T_MAX) bool
coll_types = d["coll_types"]   # (N, T_MAX) int8 (-1..3)
lengths    = d["lengths"]      # (N,)

# ball_ball(0) + 길이 20-80 에피소드 선택
rng = np.random.default_rng(7)
candidates = [i for i in range(len(lengths))
              if 25 <= lengths[i] <= 70
              and np.any(coll_flags[i, :lengths[i]-1] & (coll_types[i, :lengths[i]-1] == 0))]
ep_idx = rng.choice(candidates)

L = int(lengths[ep_idx])
ep_s = states[ep_idx, :L]           # (L, 14)
ep_f = coll_flags[ep_idx, :L-1]     # (L-1,) bool
ep_t_raw = coll_types[ep_idx, :L-1] # (L-1,) int8  원본 (-1..3)
ep_t = ep_t_raw.astype(np.int64) + 1 # v16 remap  (0..4)

T = L - 1  # rollout steps
ts = np.arange(L) * DT              # 시간축 (s)
ts_mid = (np.arange(T) + 0.5) * DT  # 구간 중앙 (충돌 표시용)

# 이름·색상
TYPE_NAMES  = ["no_coll", "ball_ball", "linear", "circular", "pocket"]
TYPE_COLORS = ["#303050", "#5588ff", "#44cc88", "#cc8844", "#ff5566"]

# ── Figure ──
fig = plt.figure(figsize=(18, 13), facecolor='#0d0f1a')
gs  = GridSpec(5, 1, figure=fig, hspace=0.06,
               top=0.93, bottom=0.06, left=0.07, right=0.97,
               height_ratios=[2.0, 1.8, 1.2, 0.7, 2.2])

title_kw = dict(color='#c8d0f0', fontsize=10, fontweight='bold',
                loc='left', pad=4)
label_kw = dict(color='#8890b0', fontsize=8.5)
spine_c  = '#2a2d45'

def style_ax(ax, ylabel='', ylim=None):
    ax.set_facecolor('#0d0f1a')
    ax.tick_params(colors='#6068a0', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(spine_c)
    if ylabel:
        ax.set_ylabel(ylabel, **label_kw)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(ts[0], ts[-1])
    ax.axhline(0, color=spine_c, lw=0.5)

# ────────────────────────────────────────────────────────────────
# Panel 0: 공 위치  (x, y)
# ────────────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.set_title(f"Episode #{ep_idx}  (L={L}, {T*DT:.2f}s)  — 실제 데이터 시간축 시각화",
              color='#c8d0f0', fontsize=12, fontweight='bold',
              loc='center', pad=6)
style_ax(ax0, ylabel='위치 (norm)')

ax0.plot(ts, ep_s[:, 0], color='#5588ff', lw=1.5, label='cue x')
ax0.plot(ts, ep_s[:, 1], color='#88aaff', lw=1.5, ls='--', label='cue y')
ax0.plot(ts, ep_s[:, 7], color='#ff8844', lw=1.5, label='tgt x')
ax0.plot(ts, ep_s[:, 8], color='#ffbb88', lw=1.5, ls='--', label='tgt y')

ax0.legend(fontsize=8, facecolor='#1a1d2e', edgecolor=spine_c,
           labelcolor='#c0c8e8', ncol=4, loc='upper right')
ax0.set_xticklabels([])

# ────────────────────────────────────────────────────────────────
# Panel 1: 속도 (vx, vy)
# ────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
style_ax(ax1, ylabel='속도 (norm)')

ax1.plot(ts, ep_s[:, 2], color='#5588ff', lw=1.2, label='cue vx')
ax1.plot(ts, ep_s[:, 3], color='#88aaff', lw=1.2, ls='--', label='cue vy')
ax1.plot(ts, ep_s[:, 9], color='#ff8844', lw=1.2, label='tgt vx')
ax1.plot(ts, ep_s[:, 10], color='#ffbb88', lw=1.2, ls='--', label='tgt vy')

ax1.legend(fontsize=8, facecolor='#1a1d2e', edgecolor=spine_c,
           labelcolor='#c0c8e8', ncol=4, loc='upper right')
ax1.set_xticklabels([])

# 충돌 구간 배경 (ax0, ax1 공유)
for t_idx in np.where(ep_f)[0]:
    t0 = t_idx * DT
    t1 = (t_idx + 1) * DT
    c  = TYPE_COLORS[ep_t[t_idx]]
    ax0.axvspan(t0, t1, alpha=0.25, color=c, lw=0)
    ax1.axvspan(t0, t1, alpha=0.25, color=c, lw=0)

# ────────────────────────────────────────────────────────────────
# Panel 2: coll_flags + coll_types (범주형 바)
# ────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
style_ax(ax2, ylabel='collision type')
ax2.set_ylim(-0.5, 4.5)
ax2.set_yticks(range(5))
ax2.set_yticklabels(TYPE_NAMES, fontsize=7.5, color='#8890b0')
ax2.set_xticklabels([])

# no_coll 배경
ax2.fill_between(ts_mid, -0.5, 4.5, color='#303050', alpha=0.15, step='mid')

# 각 스텝 충돌 타입
for t_idx in range(T):
    typ = ep_t[t_idx]
    col = TYPE_COLORS[typ]
    h   = 0.7 if ep_f[t_idx] else 0.25
    ax2.bar(t_idx * DT, h, width=DT*0.85, bottom=typ - h/2,
            color=col, alpha=0.85 if ep_f[t_idx] else 0.3, align='edge')

# 충돌 스텝 수직선
for t_idx in np.where(ep_f)[0]:
    ax2.axvline(t_idx * DT, color=TYPE_COLORS[ep_t[t_idx]], alpha=0.6, lw=1.0)

# ────────────────────────────────────────────────────────────────
# Panel 3: gt_ar 분해 레이블 (텍스트)
# ────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[3])
ax3.set_facecolor('#0d0f1a')
ax3.axis('off')
ax3.set_xlim(ts[0], ts[-1])
ax3.set_ylim(0, 1)

# 브라켓 표시
mid = ts[-1] / 2
ax3.annotate('', xy=(ts[-1]*0.465, 0.5), xytext=(ts[0], 0.5),
             arrowprops=dict(arrowstyle='-', color='#5588ff', lw=1.5))
ax3.annotate('', xy=(ts[-1]*0.935, 0.5), xytext=(ts[-1]*0.535, 0.5),
             arrowprops=dict(arrowstyle='-', color='#f0c060', lw=1.5))
ax3.text(ts[-1]*0.23, 0.82, 'ep_s[t]  →  physical GT  (14dim: x,y,vx,vy,wx,wy,wz × 2)',
         ha='center', va='center', color='#5588ff', fontsize=8.5, fontweight='bold')
ax3.text(ts[-1]*0.735, 0.82, 'one_hot(ep_t[t], 5)  →  type GT  (5dim)',
         ha='center', va='center', color='#f0c060', fontsize=8.5, fontweight='bold')
ax3.text(mid, 0.18, 'gt_ar[t]  =  cat( [physical GT (14), type one-hot (5)] )   →   19dim',
         ha='center', va='center', color='#c0c8e8', fontsize=9.5, fontweight='bold')
ax3.axvline(ts[-1]*0.5, color='#888', lw=1, ls=':')

# ────────────────────────────────────────────────────────────────
# Panel 4: gt_ar 히트맵  (19dim × T)
# ────────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[4])
ax4.set_facecolor('#0d0f1a')
for sp in ax4.spines.values():
    sp.set_color(spine_c)

# gt_ar 조립
ep_s_t = ep_s[:T]   # (T, 14)  — s_t (before-step state)
one_hot = np.zeros((T, 5), dtype=np.float32)
for t in range(T):
    one_hot[t, ep_t[t]] = 1.0
gt_ar = np.concatenate([ep_s_t, one_hot], axis=-1)  # (T, 19)

im = ax4.imshow(gt_ar.T, aspect='auto', origin='lower',
                extent=[ts[0], ts[T-1], -0.5, 18.5],
                cmap='RdYlBu_r', vmin=-1.0, vmax=1.0,
                interpolation='nearest')

# 차원 레이블
dim_labels = ['cx', 'cy', 'cvx', 'cvy', 'cwx', 'cwy', 'cwz',
              'tx', 'ty', 'tvx', 'tvy', 'twx', 'twy', 'twz',
              'nc', 'bb', 'lin', 'cir', 'pkt']
ax4.set_yticks(range(19))
ax4.set_yticklabels(dim_labels, fontsize=7.5, color='#8890b0')
ax4.set_xlabel('time (s)', color='#8890b0', fontsize=9)
ax4.tick_params(axis='x', colors='#6068a0', labelsize=8)
ax4.tick_params(axis='y', colors='#6068a0', labelsize=7.5)

# 구분선: physical vs type
ax4.axhline(13.5, color='#f0c060', lw=1.5, ls='--', alpha=0.8)
ax4.text(ts[-1]*1.005, 7,    'physical\n(14d)', va='center', color='#5588ff', fontsize=8)
ax4.text(ts[-1]*1.005, 16.5, 'type\n(5d)',      va='center', color='#f0c060', fontsize=8)

# 충돌 스텝 세로선
for t_idx in np.where(ep_f)[0]:
    ax4.axvline(t_idx * DT, color=TYPE_COLORS[ep_t[t_idx]], alpha=0.6, lw=0.8)

# 컬러바
cbar = fig.colorbar(im, ax=ax4, orientation='vertical',
                    fraction=0.015, pad=0.08, shrink=0.9)
cbar.ax.tick_params(colors='#8890b0', labelsize=7)
cbar.ax.yaxis.label.set_color('#8890b0')

ax4.set_title('gt_ar[t]  =  [ ep_s[t] (14dim) | one_hot(ep_t[t]) (5dim) ]   →   (T, 19) 히트맵',
              **title_kw)

# 범례 (충돌 타입 색상)
patches = [mpatches.Patch(color=c, label=n)
           for c, n in zip(TYPE_COLORS[1:], TYPE_NAMES[1:])]
ax4.legend(handles=patches, fontsize=7.5, facecolor='#1a1d2e',
           edgecolor=spine_c, labelcolor='#c0c8e8',
           loc='lower right', ncol=4)

plt.savefig('world_model/gt_timeseries.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"Saved → world_model/gt_timeseries.png   (ep={ep_idx}, L={L})")
plt.show()
