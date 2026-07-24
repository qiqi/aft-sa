"""Launch one solution case:
    run_solution.py <case_dir> <gpus> <model: turb|saai> [stages]

<gpus> is a comma-separated GPU index list. One index runs the legacy
single-process chain; several run the MPI chain (mpirun -np N with one rank
per GPU). L1/L2 meshes should always be run multi-GPU.
[stages]: all (default) | prep (partition+process only) | solve (solver only,
requires a prior prep with the same rank count).
"""
import os, subprocess, sys
sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai/paper/repro/driver')
sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai/paper/repro')
import saai_env

case, gpus, model = sys.argv[1], sys.argv[2], sys.argv[3]
stages = sys.argv[4] if len(sys.argv) > 4 else 'all'
nranks = len(gpus.split(','))
env = dict(os.environ)
env.update(saai_env.canonical_ai_env() if model == 'saai'
           else saai_env.classical_sa_env())
if model == 'saai' and os.environ.get('SLOWDOWN'):
    env['AI_LAMINAR_SLOWDOWN'] = os.environ['SLOWDOWN']
R = '/home/qiqi/flexcompute/compute/install/release'
env.update(OMP_NUM_THREADS='1', LD_LIBRARY_PATH=f'{R}/lib', GPU_LIST=gpus)
print(case, 'AI_SA =', env.get('AI_SA'), 'gpus', gpus, 'ranks', nranks,
      flush=True)

BIND = f'{case}/gpubind.sh'
with open(BIND, 'w') as f:
    f.write('#!/bin/bash\n'
            'IFS="," read -ra G <<< "$GPU_LIST"\n'
            'export CUDA_VISIBLE_DEVICES=${G[$OMPI_COMM_WORLD_LOCAL_RANK]}\n'
            'exec "$@"\n')
os.chmod(BIND, 0o755)


def run(tool, args, mpi, log, threads='1'):
    penv = dict(env)
    penv['OMP_NUM_THREADS'] = threads
    if mpi:
        cmd = ['mpirun', '--mca', 'btl', '^openib', '-np', str(nranks),
               './gpubind.sh', f'{R}/bin/{tool}'] + args
    else:
        penv.update(CUDA_VISIBLE_DEVICES=gpus.split(',')[0],
                    OMPI_COMM_WORLD_LOCAL_RANK='0', OMPI_COMM_WORLD_RANK='0',
                    OMPI_COMM_WORLD_SIZE='1')
        cmd = [f'{R}/bin/{tool}'] + args
    with open(f'{case}/{log}', 'w') as lf:
        rc = subprocess.run(cmd, cwd=case, env=penv,
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    print(case, tool, 'rc =', rc, flush=True)
    if rc != 0:
        sys.exit(1)


if stages in ('all', 'prep'):
    run('MeshPartitioner',
        ['--meshfile', 'mesh.cgns', '--partitions', str(nranks), '--threads', '4'],
        mpi=False, log='MeshPartitioner.log', threads='4')
    run('MeshProcessor', ['--threads', '4', 'mesh.cgns'],
        mpi=nranks > 1, log='MeshProcessor.log', threads='4')
if stages in ('all', 'solve'):
    run('Flow360Solver', ['-m', 'Flow360.json'],
        mpi=nranks > 1, log='solver_stdout.log')
