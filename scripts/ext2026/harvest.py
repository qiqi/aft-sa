"""Summarise every completed extension case on THIS host.

Emits one JSON record per case: median CL/CD/CM over the last 20% of the run,
the force drift over the final 100 rows, the converged transition fronts, and
the batch count. Run on each host; merge the JSONL streams.

  python3 harvest.py [> host.jsonl]
"""
import sys, os, csv, json

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/home/qiqi/flexcompute/flexfoil/rans",
          "/home/qiqi/flexcompute/sa-ai/paper/repro",
          "/home/qiqi/flexcompute/sa-ai/paper/repro/cfd",
          "/home/qiqi/flexcompute/sa-ai/paper/repro/driver",
          HERE):
    sys.path.insert(0, p)

import numpy as np
from run_continuation_ladders import FR
from cases_ext import CASES

host = os.uname().nodename.split(".")[0]

for c in CASES:
    wd = f"{FR}/{c['name']}"
    tf = f"{wd}/total_forces_v2.csv"
    xh = f"{wd}/xtr_history.csv"
    if not os.path.exists(tf):
        continue
    rec = dict(case=c["name"], block=c["block"], alpha=c["alpha"], re=c["re"],
               family="str" if c["name"].count("str") else "cav", host=host)
    try:
        rows = [r for r in list(csv.reader(open(tf)))[1:] if len(r) > 4]
        tail = rows[int(0.8 * len(rows)):]
        cl = np.array([float(r[2]) for r in tail])
        cd = np.array([float(r[3]) for r in tail])
        rec["CL"] = round(float(np.median(cl)), 5)
        rec["CD"] = round(float(np.median(cd)), 6)
        rec["drift"] = round(float(max(np.ptp(cl[-100:]), np.ptp(cd[-100:]))), 7)
        rec["steps"] = int(rows[-1][1])
    except Exception as e:
        rec["forces_error"] = str(e)
    if os.path.exists(xh):
        h = list(csv.DictReader(open(xh)))
        if h:
            rec["xtr_upper"] = round(float(h[-1]["xtr_upper"]), 5)
            rec["xtr_lower"] = round(float(h[-1]["xtr_lower"]), 5)
            rec["batches"] = len(h)
            if len(h) >= 2:
                rec["dxtr_upper"] = round(abs(float(h[-1]["xtr_upper"]) - float(h[-2]["xtr_upper"])), 5)
                rec["dxtr_lower"] = round(abs(float(h[-1]["xtr_lower"]) - float(h[-2]["xtr_lower"])), 5)
    print(json.dumps(rec))
