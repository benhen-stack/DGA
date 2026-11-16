"""Run DGA (Target Score) on the Hungarian Chickenpox counties.

    python scripts/run_chickenpox.py                       # all 20 counties, tuned
    python scripts/run_chickenpox.py --counties 0 5 13
    python scripts/run_chickenpox.py --no-tune --J 5 --t-max 5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dga import run_dga, tune_and_run                    # noqa: E402
from utils import all_metrics, load_chickenpox, split_windows   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counties", type=int, nargs="*", help="county indices 0..19 (default: all)")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--train-size", type=int, default=400)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="one run per seed; the mean is reported")
    ap.add_argument("--no-tune", action="store_true", help="skip validation tuning and use the flags below")
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--k-tilde", type=int, default=3)
    ap.add_argument("--J", type=int, default=5)
    ap.add_argument("--t-max", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()

    df = load_chickenpox()
    counties = a.counties if a.counties else list(range(df.shape[1]))
    fixed = dict(q=a.q, k_tilde=a.k_tilde, J=a.J, t_max=a.t_max)

    rows = []
    for c in counties:
        train, test = split_windows(df.iloc[:, c], a.window, a.train_size)
        per_seed, extra = [], {}
        for seed in a.seeds:
            if a.no_tune:
                pred, info = run_dga(train, test, fixed, seed=seed)
                info["params"] = fixed
            else:
                pred, info = tune_and_run(train, test, seed=seed)
            per_seed.append(all_metrics(test[:, -1], pred))
            extra = {"params": json.dumps(info["params"]), "n_synthetic": info["n_synthetic"]}
        m = pd.DataFrame(per_seed).mean().to_dict()
        rows.append({"county": df.columns[c], **m, **extra})
        print(f"{df.columns[c]:<10} RMSE {m['RMSE']:.3f}  MAE {m['MAE']:.3f}  {extra['params']}", flush=True)

    res = pd.DataFrame(rows)
    print("\nDGA Target Score, mean over counties")
    print(res[["MAE", "RMSE"]].mean().round(4).to_string())
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out / "chickenpox_dga.csv", index=False)
    print(f"\nSaved {a.out / 'chickenpox_dga.csv'}")


if __name__ == "__main__":
    main()
