#!/bin/bash
# Daedalus NEW-canon overnight rerun (user order 2026-07-26: finish all CFD
# except cavity L2). Self-waits for the 2D fv1 campaign to release the GPUs,
# repairs the nlf_neg results entries from the orphan harvest, then runs the
# 15 cases in three GPU phases. Each case cold-starts; run_solution.py applies
# the new canon via saai_env.canonical_ai_env().
set -u
SA=/home/qiqi/flexcompute/sa-ai
LOG=$SA/runlogs/fv1_canon_campaign2.log
RUN=$SA/daedalus/run_solution.py
cd $SA

echo "waiting for FV1-CANON-CAMPAIGN-DONE..."
until grep -q FV1-CANON-CAMPAIGN-DONE $LOG 2>/dev/null; do sleep 120; done
echo "campaign done at $(date -u +%H:%M)"

# fold the orphaned nlf_neg completions into the campaign's results JSON
python3 - << 'EOF'
import json, os
R = '/home/qiqi/flexcompute/sa-ai/flow360_fv1'
main = f'{R}/sphere_campaign_nlf_neg_results.json'
orph = main + '.orphan'
if os.path.exists(orph):
    d = json.load(open(main)) if os.path.exists(main) else {}
    o = json.load(open(orph))
    for k, v in o.items():
        if 'forces_err' in d.get(k, {'forces_err': 1}) or k not in d:
            d[k] = v
    json.dump(d, open(main, 'w'), indent=1)
    print('nlf_neg results repaired:', sorted(o))
EOF

# completeness audit of the 2D campaign
N=$(find -L $SA/flow360_fv1 -maxdepth 2 -name ai_constants.log | wc -l)
echo "2D campaign stamped cases: $N/93"

python3 $SA/scripts/stage_daedalus_fv1.py
cd $SA/daedalus_fv1

run() {  # case gpus
  echo "=== START $1 (gpus $2) $(date -u +%H:%M)"
  python3 $RUN "$1" "$2" saai > "$1/run_solution.log" 2>&1
  echo "=== DONE  $1 rc=$? $(date -u +%H:%M)"
}

echo "== PHASE 1: six L0 (1 GPU each) + ogrid L1 a4/a5 =="
run case_ogrid_L0_saai_a4  0 &
run case_ogrid_L0_saai_a5  1 &
run case_ogrid_L0_saai_a6  2 &
run case_cavity_L0_saai_a4 3 &
run case_cavity_L0_saai_a5 4 &
run case_cavity_L0_saai_a6 5 &
run case_ogrid_L1_saai_a4  6 &
run case_ogrid_L1_saai_a5  7 &
wait

echo "== PHASE 2: ogrid L1 a6 + three cavity L1 (2 GPUs each) =="
run case_ogrid_L1_saai_a6  0 &
run case_cavity_L1_saai_a4 1,2 &
run case_cavity_L1_saai_a5 3,4 &
run case_cavity_L1_saai_a6 5,6 &
wait

echo "== PHASE 3: ogrid L2 sequential (6 ranks) =="
for a in 4 5 6; do
  run case_ogrid_L2_saai_a$a 0,1,2,3,4,5
done
echo "DAEDALUS-FV1-DONE"
