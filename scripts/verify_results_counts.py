import pandas as pd

def check(path, exp_seq, exp_tips, label):
    df = pd.read_csv(path)
    print("="*78)
    print(f"{label}: {path}")
    print(f"  columns with 'tip': {[c for c in df.columns if 'tip' in c.lower()]}")
    for split in ["test"]:
        d = df[df["split"] == split]
        print(f"  -- split={split} | expect {exp_seq} seqs, {exp_tips} tips per (method,run) --")
        print(f"  {'method':<12}{'run':>4}{'n_seq':>8}{'n_tips':>9}{'  OK?'}")
        bad = []
        for (m, r), g in d.groupby(["method", "run"]):
            nseq = g["sequence"].nunique()
            ntips = int(g["n_tips_evaluated"].sum())
            ok = (nseq == exp_seq) and (ntips == exp_tips)
            if not ok:
                bad.append((m, r, nseq, ntips))
            print(f"  {m:<12}{int(r):>4}{nseq:>8}{ntips:>9}{'   ' + ('ok' if ok else 'MISMATCH')}")
        if bad:
            print(f"  !! {len(bad)} mismatching (method,run) groups")
        else:
            print("  ALL GROUPS OK")

check("output/tables/results_test_organs_epoch15000.csv", 45, 189, "ORGANS")
check("output/tables/results_test_plants_epoch15000.csv", 8, 192, "PLANTS")
