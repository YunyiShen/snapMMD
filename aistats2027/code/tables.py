"""Recompute every cell of every results table of the paper from results/scores.csv and compare it with the printed value.
The paper's table sources are in paper_tables/ (tables/*.tex of the paper, and the six per-task tables of App. D).
Writes results/recomputed_tables.csv (one row per cell: table, block, row, column, printed and recomputed mean and SD)
and results/check_report.txt (cells that differ at the printed precision). Run after score.py."""
import csv, glob, os, re
from collections import defaultdict
import numpy as np
from common import ROOT

S = defaultdict(list)          # (method, task, kind, t, metric) -> list over seeds
for r in csv.DictReader(open(f"{ROOT}/results/scores.csv")):
    for metric in ["mmd2", "mmd2_snap/4", "mmd2_4snap", "mmd2_h_train", "emd"]:
        S[(r["method"], r["task"], r["kind"], int(r["t"]), metric)].append(float(r[metric]))
TASKS = ["LV", "ReprParam", "ReprSemiparam", "ReprProtein", "GoM", "PBMC"]
TIMEFILES = {"LV_classic": "LV", "Repressilator_classic": "ReprParam", "Repressilator_mlp": "ReprSemiparam",
             "Repressilator_missingobs": "ReprProtein", "GoM_realdata": "GoM", "pbmc_realdata": "PBMC"}


def method_of(label):
    """Row label of a table -> method name of scores.csv."""
    s = re.sub(r"\\textbf\{([^}]*)\}|\\texttt\{([^}]*)\}", lambda m: m.group(1) or m.group(2), label).strip()
    s = s.replace("$\\sigma$", "sigma").replace("(interpolation)", "").replace("(forecast)", "").strip()
    return s


def values(method, task, kind, metric, t=None):
    if t is None:
        keys = [k for k in S if k[0] == method and k[1] == task and k[2] == kind and k[4] == metric]
        v = [x for k in keys for x in S[k]]
    else:
        v = S.get((method, task, kind, t, metric), [])
    return np.array(v)


def cells(line):
    """Split a table row into (label, [(mean_str, sd_str or None) or None for '--'])."""
    parts = [p.strip() for p in re.sub(r"\\\\\s*$", "", line).split("&")]
    out = []
    for p in parts[1:]:
        p = re.sub(r"\\cellcolor\{[^}]*\}", "", p)
        m = re.search(r"\\ms\{([-\d.]+)\}\{([-\d.]+)\}", p)
        if m: out.append((m.group(1), m.group(2)))
        else:
            n = re.search(r"(-?\d+(?:\.\d+)?)", p); out.append((n.group(1), None) if n else None)
    return parts[0], out


def rnd(x, s):
    dec = len(s.split(".")[1]) if "." in s else 0
    return f"{x:.{dec}f}"


rows, bad = [], []
ALLTIMES = {}
def check(table, block, row, col, printed, v, ratio=None):
    if printed is None:
        return
    if ratio is not None:
        rec_m = ratio; rec_sd = None
    elif len(v) == 0:
        rows.append([table, block, row, col, printed[0], printed[1], "missing", ""]); bad.append(rows[-1]); return
    else:
        rec_m, rec_sd = v.mean(), (v.std() if printed[1] is not None else None)
    m_ok = rnd(rec_m, printed[0]) == printed[0] or abs(rec_m - float(printed[0])) <= 0.5 * 10 ** -(len(printed[0].split(".")[1]) if "." in printed[0] else 0) * 1.0001
    sd_ok = printed[1] is None or rec_sd is None or rnd(rec_sd, printed[1]) == printed[1] or abs(rec_sd - float(printed[1])) <= 0.5 * 10 ** -len(printed[1].split(".")[1]) * 1.0001
    rows.append([table, block, row, col, printed[0], printed[1] or "", f"{rec_m:.6g}", "" if rec_sd is None else f"{rec_sd:.6g}"])
    if not (m_ok and sd_ok): bad.append(rows[-1])


for path in sorted(glob.glob(f"{ROOT}/paper_tables/*.tex")):
    name = os.path.basename(path)[:-4]; lines = open(path).read().split("\n")
    ALLTIMES[name] = [t for l in lines if l.strip().startswith(r"\textbf{Time}") for t in re.findall(r"\\textbf\{(\d+(?:\.\d+)?)\}", l)]
    header, block, bw = None, "", None
    for line in lines:
        s = line.strip()
        if s.startswith(r"\textbf{Forecast}"): block = "forecast"
        if s.startswith(r"\par\medskip\textbf{Interpolation}"): block = "interp"
        if r"\textit{Bandwidth" in s:
            bw = "mmd2_h_train" if "mathrm{train}" in s else "mmd2_snap/4" if "h_t/4" in s else "mmd2_4snap" if "4h_t" in s else "mmd2"
        if s.startswith(r"\textbf{Method}") or s.startswith(r"\textbf{Time}") or (s.startswith("& ") or s.startswith(" & ")):
            header = [re.sub(r"\\textcolor\{red\}\{|\\textbf\{|\}|\$\^2\$", "", h).strip() for h in s.rstrip("\\ ").split("&")[1:]]; continue
        if header is None or "&" not in s or s.startswith("%") or s.startswith(r"\multicolumn"): continue
        label, cs = cells(s)
        if name.startswith("summary_combined_mmd"):
            for c, p in zip(header, cs): check(name, "", label, c, p, values(method_of(label), c, "forecast", "mmd2"))
        elif name in ("summary_interpolation_mmd", "summary_interpolation_emd"):
            met = "mmd2" if name.endswith("mmd") else "emd"
            for c, p in zip(header, cs): check(name, "", label, c, p, values(method_of(label), c, "interp", met))
        elif name == "summary_combined_emd":
            for c, p in zip(header, cs): check(name, block, label, c, p, values(method_of(label), c, block, "emd"))
        elif name.rsplit("_", 1)[0] in TIMEFILES:
            task = TIMEFILES[name.rsplit("_", 1)[0]]; met = "mmd2" if name.endswith("MMD") else "emd"
            for c, p in zip(header, cs): check(name, "", label, c, p, values(method_of(label), task, "interp", met, t=ALLTIMES[name].index(c)))
        elif name.startswith("per_task_"):
            task = name[len("per_task_"):]
            for c, p in zip(header, cs):
                if c.startswith("Forecast-MMD"): check(name, "", label, c, p, values(method_of(label), task, "forecast", "mmd2"))
                elif c.startswith("Forecast-EMD"): check(name, "", label, c, p, values(method_of(label), task, "forecast", "emd"))
        elif name == "mmd_bandwidth_sweep":
            kind = "interp" if "(interpolation)" in label else "forecast"
            for c, p in zip(header, cs): check(name, bw, label, c, p, values(method_of(label), c, kind, bw))
        elif name.startswith("abl_"):
            kind = "forecast" if name.endswith("forecast") else "interp"
            meth = "Fixed volatility" if "fixed" in name else "Fully neural"
            src = {"SnapMMD, mean $\\pm$ SD": "Ours", f"{meth}, mean $\\pm$ SD": meth}
            for c, p in zip(header, cs):
                if label in src: check(name, "", label, c, p, values(src[label], c, kind, "mmd2"))
                elif label == "Ratio":
                    a, o = values(meth, c, kind, "mmd2"), values("Ours", c, kind, "mmd2")
                    if len(a) and len(o): check(name, "", label, c, p, None, ratio=a.mean() / o.mean())

os.makedirs(f"{ROOT}/results", exist_ok=True)
with open(f"{ROOT}/results/recomputed_tables.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["table", "block", "row", "column", "printed_mean", "printed_sd", "recomputed_mean", "recomputed_sd"]); w.writerows(rows)
with open(f"{ROOT}/results/check_report.txt", "w") as fh:
    fh.write(f"{len(rows)} table cells checked; {len(bad)} differ from the paper at the printed precision.\n")
    for b in bad: fh.write(" | ".join(map(str, b)) + "\n")
print(f"{len(rows)} cells checked, {len(bad)} differ (results/check_report.txt)")
