"""Sphere-kernel airfoil campaign: fresh-converge each canonical case IN PLACE
in flow360_fr (reuse existing meshes; figures read SAAI_CFD_ROOT=flow360_fr),
4 concurrent on GPUs 0-3 ONLY (agent on 4-7).

Per case: reset solution state (keep mesh) -> converge_by_xtr (current sphere
binary, AI_SA=1, AI_LAMINAR_SLOWDOWN=0.01, default c_nu,ai=1/12) -> record
CL/CD/x_tr -> delete solver.log to stay light on disk. Post-proc
(add_derived_to_slice) and figure regen are done separately.

Usage: python3 run_sphere_campaign.py <set>   set in {nlf, eppler, epp_sweep}
"""
import os, sys, json, shutil, subprocess, threading, queue, time, glob, csv

FR = "/home/qiqi/flexcompute/sa-ai/flow360_fr"
AI = "/home/qiqi/flexcompute/sa-ai/flow360_ai"
PP = ("/home/qiqi/flexcompute/sa-ai/paper/repro/cfd:"
      "/home/qiqi/flexcompute/sa-ai/paper/repro:"
      "/home/qiqi/flexcompute/sa-ai/paper")
GPUS = [0, 1, 2, 3]

SETS = {
    'nlf': [f"{m}{L}prop_nlf0416_Re4M_a{a}"
            for m in ('cav', 'str') for L in ('L0', 'L1', 'L2')
            for a in (0, 4, 9, 15)],
    'eppler': [f"{m}{L}prop_eppler387_Re200k_a{a}"
               for m in ('cav', 'str') for L in ('L0', 'L1', 'L2')
               for a in (0, 2, 5, 7)],
    'epp_sweep': ([f"sweep_Re{Rk}k_a5" for Rk in (60, 100, 300, 460)]
                  + [f"sweep_str_Re{Rk}k_a5" for Rk in (60, 100, 300, 460)]),
}

RM_GLOBS = ['*_v2.csv', '*.pvtu', '*.vtu', 'xtr_history.csv',
            'restart_rank_*.dmp', 'restart.json', 'solver.log',
            'postprocessor.log', 'timer.json', 'progress.csv',
            'ipc_control.sock', 'converge_fr.log', 'slice_with_derived_*.vtu']


def reset(d):
    """Delete solution/output state; keep mesh.cgns*, *.json config. Fresh start."""
    for pat in RM_GLOBS:
        for f in glob.glob(f"{d}/{pat}"):
            try:
                os.remove(f)
            except OSError:
                pass
    shutil.rmtree(f"{d}/restartOutput", ignore_errors=True)
    shutil.rmtree(f"{d}/ipc_data", ignore_errors=True)
    p = f"{d}/Flow360.json"
    c = json.load(open(p))
    c['runControl']['restart'] = False
    c.setdefault('timeStepping', {})['maxPseudoSteps'] = 5000
    json.dump(c, open(p, 'w'), indent=1)


def parse(d):
    out = {}
    try:
        tr = list(csv.reader(open(f"{d}/total_forces_v2.csv")))
        h = [x.strip() for x in tr[0]]
        last = [x for x in tr[-1] if x.strip() != '']
        out['CL'] = round(float(last[h.index('CL')]), 4)
        out['CD'] = round(float(last[h.index('CD')]), 5)
    except Exception as e:
        out['forces_err'] = str(e)
    try:
        hr = list(csv.DictReader(open(f"{d}/xtr_history.csv")))
        out['xtr_up'] = round(float(hr[-1]['xtr_upper']), 4)
        out['xtr_lo'] = round(float(hr[-1]['xtr_lower']), 4)
        out['steps'] = int(hr[-1]['step'])
    except Exception as e:
        out['xtr_err'] = str(e)
    return out


def worker(gpu, q, results, lock):
    env = dict(os.environ)
    env['PYTHONPATH'] = PP
    while True:
        try:
            name = q.get_nowait()
        except queue.Empty:
            return
        d = f"{FR}/{name}"
        if not os.path.exists(f"{d}/mesh.cgns"):
            with lock:
                results[name] = {'err': 'no mesh'}
            print(f"[gpu{gpu}] SKIP {name} (no mesh)", flush=True)
            q.task_done()
            continue
        t0 = time.time()
        print(f"[gpu{gpu}] START {name}", flush=True)
        reset(d)
        log = open(f"/tmp/camp_{name}.log", "w")
        rc = subprocess.run(
            ['python3', f"{AI}/converge_by_xtr.py", d,
             '--batch', '5000', '--gpu', str(gpu),
             '--max-batches', '16', '--min-batches', '2', '--tol', '0.01'],
            env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        log.close()
        conv = 'CONVERGED' in open(f"/tmp/camp_{name}.log").read()
        r = parse(d)
        r['converged'] = conv
        r['dt_min'] = round((time.time() - t0) / 60, 1)
        # stay light on disk: solver.log can be ~100 MB
        for f in glob.glob(f"{d}/solver.log"):
            try:
                os.remove(f)
            except OSError:
                pass
        with lock:
            results[name] = r
        print(f"[gpu{gpu}] DONE {name}: {r}", flush=True)
        q.task_done()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'nlf'
    cases = SETS[which]
    print(f"=== campaign '{which}': {len(cases)} cases on GPUs {GPUS} ===", flush=True)
    q = queue.Queue()
    for c in cases:
        q.put(c)
    results = {}
    lock = threading.Lock()
    ts = [threading.Thread(target=worker, args=(g, q, results, lock)) for g in GPUS]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    out = f"{FR}/sphere_campaign_{which}_results.json"
    json.dump(results, open(out, 'w'), indent=1)
    nconv = sum(1 for r in results.values() if r.get('converged'))
    print(f"\n=== {which} DONE: {nconv}/{len(cases)} converged -> {out} ===", flush=True)
    for k in sorted(results):
        print(f"  {k}: {results[k]}", flush=True)


if __name__ == '__main__':
    main()
