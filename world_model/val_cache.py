"""
Val episodes 캐시 저장/로드.
최초 1회 실행으로 val_eps_cache.npz 생성 → 이후 로드는 ~0.5초.

전체 데이터셋 캐시:
  save_dataset_cache / load_dataset_cache — np.savez (비압축) 사용, 로딩 ~1초.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

CACHE_PATH = Path('world_model/val_eps_cache.npz')


def _save_episodes(episodes: list, path: Path, compress: bool):
    N       = len(episodes)
    max_len = max(len(e[0]) for e in episodes)

    states  = np.zeros((N, max_len, 14), dtype=np.float32)
    flags   = np.zeros((N, max_len - 1), dtype=bool)
    types   = np.zeros((N, max_len - 1), dtype=np.int64)
    lengths = np.zeros(N, dtype=np.int64)
    n_cushs = np.zeros(N, dtype=np.int64)

    for i, (ep_s, ep_f, ep_t, n_cush) in enumerate(episodes):
        L = len(ep_s)
        states[i, :L]   = ep_s
        flags[i,  :L-1] = ep_f
        types[i,  :L-1] = ep_t
        lengths[i]       = L
        n_cushs[i]       = n_cush

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(path, states=states, flags=flags, types=types,
            lengths=lengths, n_cushs=n_cushs)
    print(f"Saved cache → {path}  ({N} eps, max_len={max_len}, compress={compress})")


def _load_episodes(path: Path, copy: bool = True) -> list:
    d = np.load(path)
    eps = []
    for i, L in enumerate(d['lengths']):
        L = int(L)
        s, f, t = d['states'][i, :L], d['flags'][i, :L-1], d['types'][i, :L-1]
        eps.append((s.copy() if copy else s,
                    f.copy() if copy else f,
                    t.copy() if copy else t,
                    int(d['n_cushs'][i])))
    return eps


def save_val_cache(val_eps: list, path: Path = CACHE_PATH):
    _save_episodes(val_eps, path, compress=True)


def load_val_cache(path: Path = CACHE_PATH) -> list:
    return _load_episodes(path)


def save_dataset_cache(episodes: list, path: Path):
    """전체 학습 데이터 캐시 — pickle로 저장 (패딩 없음, ~200MB)."""
    import pickle
    pkl_path = Path(str(path).replace('.npz', '.pkl'))
    with open(pkl_path, 'wb') as f:
        pickle.dump(episodes, f, protocol=4)
    total_mb = sum(e[0].nbytes + e[1].nbytes + e[2].nbytes for e in episodes) / 1024**2
    print(f"Saved dataset cache → {pkl_path}  ({len(episodes):,} eps, ~{total_mb:.0f}MB)")


def load_dataset_cache(path: Path) -> list:
    import pickle
    pkl_path = Path(str(path).replace('.npz', '.pkl'))
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def dataset_cache_exists(path: Path) -> bool:
    pkl_path = Path(str(path).replace('.npz', '.pkl'))
    return pkl_path.exists()


if __name__ == '__main__':
    from world_model.train_ssm import SSMDataset, make_balanced_val_eps
    ds      = SSMDataset('world_model/data_fixeddt')
    val_eps = make_balanced_val_eps(ds.episodes, n_each=250, seed=0)
    save_val_cache(val_eps, CACHE_PATH)
