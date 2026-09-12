"""bb vs nbb 에러 비교 플랏 — v14 / v15 / v16"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec

for fname in ['AppleGothic', 'NanumGothic']:
    try:
        fm.findfont(fm.FontProperties(family=fname), fallback_to_default=False)
        plt.rcParams['font.family'] = fname
        break
    except Exception:
        continue

# ── 로그 파서 ──────────────────────────────────────────────────
PAT = re.compile(
    r'Epoch\s+(\d+)/\d+\s+\[(\S+)\s.*?'
    r'err=([\d.]+)cm\s+\(bb=([\d.]+)/nbb=([\d.]+)\)'
)
STAGE_PAT = re.compile(r'\*\*\* Curriculum → Stage \d+: (\S+) \*\*\*')

def parse_log(path):
    epochs, bb, nbb, err, stages = [], [], [], [], []
    cur_stage = 'P1a'
    with open(path) as f:
        for line in f:
            m = STAGE_PAT.search(line)
            if m:
                cur_stage = m.group(1)
            m = PAT.search(line)
            if m:
                ep   = int(m.group(1))
                stage = m.group(2)   # label from bracket
                epochs.append(ep)
                err.append(float(m.group(3)))
                bb.append(float(m.group(4)))
                nbb.append(float(m.group(5)))
                stages.append(stage)
    return (np.array(epochs), np.array(bb), np.array(nbb),
            np.array(err), stages)

logs = {
    'v14': 'world_model/results/ssm_v14_ss/train_20260909_134955.log',
    'v15': 'world_model/results/ssm_v15_ar/train_20260910_072018.log',
    'v16': 'world_model/results/ssm_v16_5cls/train_20260910_202850.log',
}

data = {k: parse_log(v) for k, v in logs.items()}

# ── 색상 ──────────────────────────────────────────────────────
VC = {'v14': '#5588ff', 'v15': '#ff8844', 'v16': '#44cc88'}

# ── Figure ────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11), facecolor='#0d0f1a')
gs  = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.28,
               top=0.93, bottom=0.07, left=0.07, right=0.97)

spine_c = '#2a2d45'

def style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#0d0f1a')
    ax.tick_params(colors='#6068a0', labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_color(spine_c)
    if title:
        ax.set_title(title, color='#c8d0f0', fontsize=10, fontweight='bold', pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, color='#8890b0', fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color='#8890b0', fontsize=8.5)

# ── [0,0] bb err 비교 ────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
style(ax, 'bb (ball-ball) error  per epoch', 'epoch', 'error (cm)')
for v, (ep, bb, nbb, err, st) in data.items():
    ax.plot(ep, bb, color=VC[v], lw=1.2, alpha=0.85, label=v)
ax.legend(fontsize=9, facecolor='#1a1d2e', edgecolor=spine_c, labelcolor='#c0c8e8')
ax.axhline(50, color='#555', lw=0.7, ls='--')

# ── [0,1] nbb err 비교 ────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
style(ax, 'nbb (no ball-ball) error  per epoch', 'epoch', 'error (cm)')
for v, (ep, bb, nbb, err, st) in data.items():
    ax.plot(ep, nbb, color=VC[v], lw=1.2, alpha=0.85, label=v)
ax.legend(fontsize=9, facecolor='#1a1d2e', edgecolor=spine_c, labelcolor='#c0c8e8')

# ── [1,0] bb-nbb gap ─────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
style(ax, 'bb - nbb  gap  (클수록 충돌 예측 어려움)', 'epoch', 'gap (cm)')
for v, (ep, bb, nbb, err, st) in data.items():
    ax.plot(ep, bb - nbb, color=VC[v], lw=1.2, alpha=0.85, label=v)
ax.axhline(0, color='#555', lw=0.7, ls='--')
ax.legend(fontsize=9, facecolor='#1a1d2e', edgecolor=spine_c, labelcolor='#c0c8e8')

# ── [1,1] overall err 비교 ────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
style(ax, 'overall err  per epoch', 'epoch', 'error (cm)')
for v, (ep, bb, nbb, err, st) in data.items():
    ax.plot(ep, err, color=VC[v], lw=1.2, alpha=0.85, label=v)
ax.legend(fontsize=9, facecolor='#1a1d2e', edgecolor=spine_c, labelcolor='#c0c8e8')

# ── [2, :] v16 스테이지별 bb/nbb 상세 ──────────────────────
ax = fig.add_subplot(gs[2, :])
style(ax, 'v16  bb vs nbb  error  (stage 표시)', 'epoch', 'error (cm)')

ep16, bb16, nbb16, err16, st16 = data['v16']
ax.plot(ep16, bb16,  color='#ff6666', lw=1.3, label='v16 bb',  alpha=0.9)
ax.plot(ep16, nbb16, color='#66aaff', lw=1.3, label='v16 nbb', alpha=0.9)
ax.fill_between(ep16, bb16, nbb16, alpha=0.12, color='#ffaa66')

# 스테이지 전환 세로선 + 레이블
prev_st = st16[0]
stage_starts = []
for i, s in enumerate(st16):
    if s != prev_st:
        stage_starts.append((ep16[i], s))
        prev_st = s

for x_ep, sname in stage_starts:
    col = '#f0c060' if 'C' in sname else ('#ff7070' if 'R' in sname else '#7070f0')
    ax.axvline(x_ep, color=col, lw=0.8, ls='--', alpha=0.6)
    ax.text(x_ep + 0.5, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100,
            sname, color=col, fontsize=6.5, va='top', rotation=90, alpha=0.85)

ax.legend(fontsize=9, facecolor='#1a1d2e', edgecolor=spine_c, labelcolor='#c0c8e8')

# ── best 수치 요약 (박스) ─────────────────────────────────────
summary = {}
for v, (ep, bb, nbb, err, st) in data.items():
    # 마지막 10에폭 평균 (수렴 후)
    summary[v] = {
        'bb_min':  bb.min(),
        'nbb_min': nbb.min(),
        'bb_last': bb[-10:].mean(),
        'nbb_last': nbb[-10:].mean(),
    }

txt = "  최저 bb err         최저 nbb err        최근 10ep bb        최근 10ep nbb\n"
for v in ['v14', 'v15', 'v16']:
    s = summary[v]
    txt += (f"  {v}: {s['bb_min']:.1f}cm"
            f"                {s['nbb_min']:.1f}cm"
            f"                {s['bb_last']:.1f}cm"
            f"                {s['nbb_last']:.1f}cm\n")

fig.text(0.07, 0.005, txt, color='#c0c8e8', fontsize=8.5,
         fontfamily='monospace', va='bottom')

fig.suptitle('v14 / v15 / v16  —  bb vs nbb 에러 비교', color='#c8d0f0',
             fontsize=13, fontweight='bold')

plt.savefig('world_model/bb_err_compare.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved → world_model/bb_err_compare.png")

# 수치 출력
print("\n=== 최저 bb/nbb 에러 ===")
for v in ['v14', 'v15', 'v16']:
    ep, bb, nbb, err, st = data[v]
    print(f"{v}:  bb_min={bb.min():.1f}cm (ep{ep[bb.argmin()]})  "
          f"nbb_min={nbb.min():.1f}cm (ep{ep[nbb.argmin()]})")
print("\n=== 최근 10에폭 평균 ===")
for v in ['v14', 'v15', 'v16']:
    ep, bb, nbb, err, st = data[v]
    print(f"{v}:  bb={bb[-10:].mean():.1f}cm  nbb={nbb[-10:].mean():.1f}cm  "
          f"gap={bb[-10:].mean()-nbb[-10:].mean():.1f}cm")

plt.show()
