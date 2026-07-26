"""Stage the Daedalus NEW-canon rerun (user order 2026-07-26: all CFD except
unstructured/cavity L2): clone the 15 case dirs (L0+L1 both families, L2
o-grid) into /local_data/qiqi/sa-ai/daedalus_fv1 with mesh files HARDLINKED
(same filesystem) and all prior outputs excluded, so each case cold-starts
under the new canon (AI_FV1BYPASS rides in via saai_env.canonical_ai_env()
inside run_solution.py).
"""
import json
import os
import shutil

SRC = '/home/qiqi/flexcompute/sa-ai/daedalus'
DST_REAL = '/local_data/qiqi/sa-ai/daedalus_fv1'
DST_LINK = '/home/qiqi/flexcompute/sa-ai/daedalus_fv1'

CASES = ([f'case_{fam}_L{l}_saai_a{a}' for fam in ('ogrid', 'cavity')
          for l in (0, 1) for a in (4, 5, 6)] +
         [f'case_ogrid_L2_saai_a{a}' for a in (4, 5, 6)])

KEEP_EXACT = {'Flow360.json', 'Flow360Mesh.json'}
KEEP_PREFIX = ('mesh.cgns',)          # mesh + partitioner data, hardlinked
os.makedirs(DST_REAL, exist_ok=True)
if not os.path.islink(DST_LINK) and not os.path.exists(DST_LINK):
    os.symlink(DST_REAL, DST_LINK)

for c in CASES:
    s = os.path.realpath(f'{SRC}/{c}')
    d = f'{DST_REAL}/{c}'
    os.makedirs(d, exist_ok=True)
    if os.path.exists(f'{d}/STAGED'):
        print(f'SKIP {c} (staged)')
        continue
    # mesh files may live in the a5 sibling (hardlink pattern of the old runs)
    mesh_src = s if os.path.exists(f'{s}/mesh.cgns') else \
        os.path.realpath(f"{SRC}/{c.rsplit('_a', 1)[0]}_a5")
    n = 0
    for f in os.listdir(mesh_src):
        if f.startswith(KEEP_PREFIX) and not f.endswith('.log'):
            if not os.path.exists(f'{d}/{f}'):
                os.link(f'{mesh_src}/{f}', f'{d}/{f}')
            n += 1
    for f in KEEP_EXACT:
        shutil.copy2(f'{s}/{f}', f'{d}/{f}')
    j = json.load(open(f'{d}/Flow360.json'))
    j.setdefault('runControl', {})['restart'] = False
    json.dump(j, open(f'{d}/Flow360.json', 'w'), indent=4)
    open(f'{d}/STAGED', 'w').write('2026-07-26 new-canon rerun\n')
    print(f'STAGED {c} ({n} mesh files hardlinked from '
          f'{os.path.basename(mesh_src)})')
print('DAEDALUS-FV1-STAGED')
