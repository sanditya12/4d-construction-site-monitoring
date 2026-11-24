import pickle, numpy as np

logs = pickle.load(open("/home_domuser2/s84klaks/src/4d-construction-site-monitoring/data/outputs/sfm/loc_july_in_may/t2_poses.txt_logs.pkl", "rb"))

stats = {}
for q, d in logs["loc"].items():
    r = d["PnP_ret"]
    if r is None:                      # some versions store None for failures
        continue
    n = int(r["num_inliers"])
    m = len(np.asarray(r["inlier_mask"]))
    stats[q] = (n, n / max(m, 1))      # inlier count and inlier ratio

counts = np.array([n for n, _ in stats.values()])
for thr in (10, 20, 30, 50):
    print(f"min inliers {thr}: keeps {(counts >= thr).sum()} / {len(stats)}")

MIN_INLIERS = 30
keep = {q for q, (n, _) in stats.items() if n >= MIN_INLIERS}
with open("/home_domuser2/s84klaks/src/4d-construction-site-monitoring/data/outputs/sfm/loc_july_in_may/t2_poses.txt") as f, open("/home_domuser2/s84klaks/src/4d-construction-site-monitoring/data/outputs/sfm/loc_july_in_may/t2_poses_filtered.txt", "w") as g:
    for line in f:
        if line.split()[0] in keep:
            g.write(line)