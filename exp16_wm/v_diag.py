"""V diagnosis: does PRm i4's V head tell which positions lead to a run-out? Greedy games (eval seed 950, not used anywhere else),
target(s_t) = every shot from t to the end of the game pocketed a ball without a scratch. Compared against trivial baselines."""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/yj/Documents/billiards-rl")
from sklearn.metrics import roc_auc_score

from exp16_wm.bc import TransformedPolicy
from exp16_wm.made_ensemble import load_ensemble
from exp16_wm.obs_transform import n_balls_raw, relative_obs
from exp16_wm.teacher_data import clear_env_kw
from simulator import BilliardsEnv

torch.set_num_threads(4)
run = glob.glob("/Users/yj/Documents/billiards-rl/logs/experiments_cloud/pr-iter-1/out/PR/*_i4_*_s0_*")[0]
ens = load_ensemble([run], "mean")
pol = TransformedPolicy(ens, "relative")
env = BilliardsEnv(**clear_env_kw(6)); env.reset(seed=950)
obs_l, y_l, g_l = [], [], []
for g in range(int(sys.argv[1]) if len(sys.argv) > 1 else 300):
    obs, _ = env.reset(); done = False; ok = []; st = []
    while not done:
        st.append(np.asarray(obs, np.float32))
        obs, _, term, trunc, info = env.step(pol.act(obs, deterministic=True))
        done = term or trunc
        ok.append(int(info["pocketed_this_step"]) > 0 and not info.get("scratch", False))
    ok = np.array(ok)
    tail = np.flip(np.cumprod(np.flip(ok)))                      # tail[t] = all shots from t on succeeded
    obs_l += st; y_l += list(tail.astype(int)); g_l += [g] * len(st)
X = np.stack(obs_l); y = np.array(y_l)
rel = torch.as_tensor(relative_obs(X))
m = ens.models[0]
with torch.no_grad():
    o = m.forward_full(rel)
    v = torch.sigmoid(m.v_logit(o)).numpy()
    z, _ = ens.scores(rel)
    made_max = torch.sigmoid(z.reshape(len(X), -1).max(1).values).numpy()
nb = n_balls_raw(X.shape[1])
left = nb - X[:, 2 + 2:2 + 3 * nb:3].sum(1)
print(f"states {len(X)}, games {g_l[-1] + 1}, positive rate {y.mean():.1%}")
print(f"V        AUC {roc_auc_score(y, v):.3f}   mean V {v.mean():.3f} (calibration: true {y.mean():.3f})")
print(f"−balls left AUC {roc_auc_score(y, -left):.3f}")
print(f"made max AUC {roc_auc_score(y, made_max):.3f}")
for k in sorted(set(left.astype(int))):
    s = left == k
    if s.sum() > 30 and 0 < y[s].mean() < 1:
        print(f"  balls left {k}: n {s.sum()}, true {y[s].mean():.2f}, V mean {v[s].mean():.2f}, V AUC {roc_auc_score(y[s], v[s]):.3f}, "
              f"made-max AUC {roc_auc_score(y[s], made_max[s]):.3f}")
