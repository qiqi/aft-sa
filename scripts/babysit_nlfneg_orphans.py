"""Babysit the two orphaned nlf_neg converge_by_xtr children (their parent
was killed to prevent a duplicate-launch collision with the main campaign's
nlf_neg set, 2026-07-25 22:03 UTC). When each exits: stamp ai_constants.log,
harvest forces + final xtr, and merge into the campaign's nlf_neg results
JSON (repairing the 'err' entries the campaign recorded when its duplicate
launches died against the held case dirs).
"""
import csv, json, os, sys, time

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/cfd")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from run_continuation_ladders import forces, write_ai_constants  # noqa: E402

ROOT = "/home/qiqi/flexcompute/sa-ai/flow360_fv1"
WATCH = {1308952: "cavL2prop_nlf0416_Re4M_am8",
         1674569: "strL2prop_nlf0416_Re4M_am4"}
RESJ = f"{ROOT}/sphere_campaign_nlf_neg_results.json"


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def harvest(tag):
    wd = f"{ROOT}/{tag}"
    out = {}
    try:
        f = forces(wd)
        out.update({'CL': f['CL'], 'CD': f['CD']})
    except Exception as e:
        out['forces_err'] = str(e)[:80]
    try:
        hr = list(csv.DictReader(open(f"{wd}/xtr_history.csv")))
        out['xtr_up'] = round(float(hr[-1]['xtr_upper']), 4)
        out['xtr_lo'] = round(float(hr[-1]['xtr_lower']), 4)
        if len(hr) >= 2:
            du = abs(float(hr[-1]['xtr_upper']) - float(hr[-2]['xtr_upper']))
            dl = abs(float(hr[-1]['xtr_lower']) - float(hr[-2]['xtr_lower']))
            out['front_converged'] = bool(du < 0.01 and dl < 0.01)
    except Exception as e:
        out['xtr_err'] = str(e)[:80]
    try:
        write_ai_constants(wd)
        out['stamped'] = True
    except Exception as e:
        out['stamp_err'] = str(e)[:80]
    return out


pending = dict(WATCH)
while pending:
    time.sleep(60)
    for pid in list(pending):
        if alive(pid):
            continue
        tag = pending.pop(pid)
        time.sleep(10)  # let final writes land
        r = harvest(tag)
        print(f"ORPHAN-DONE {tag}: {r}", flush=True)
        try:
            d = json.load(open(RESJ)) if os.path.exists(RESJ) else {}
        except Exception:
            d = {}
        d[tag] = r
        json.dump(d, open(RESJ + '.orphan', 'w'), indent=1)
print("NLFNEG-ORPHANS-DONE", flush=True)
