"""Real PostgreSQL tests in a new disposable Unix-socket-only cluster."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path('/tmp/quant-v3-pg-8VsqEo/root')
BIN = RUNTIME / 'usr/lib/postgresql/16/bin'
OUT = ROOT / 'evidence/v8/postgres_full'

def main():
    OUT.mkdir(exist_ok=False)
    temporary = Path(tempfile.mkdtemp(prefix='quant-v8-pg-'))
    data = temporary / 'data'
    env = dict(os.environ,
        LD_LIBRARY_PATH=str(RUNTIME / 'usr/lib/x86_64-linux-gnu'),
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', POLARS_MAX_THREADS='2',
        PYTHONPATH='/tmp/quant-v3-pg-8VsqEo/driver:factor_optimizer:factor_preprocess:.')
    started = False
    result = None
    with (OUT / 'server.log').open('w') as log:
        def command(args):
            return subprocess.run([str(x) for x in args],env=env,stdout=log,stderr=subprocess.STDOUT,
                                  check=True,timeout=60)
        try:
            command([BIN/'initdb','-D',data,'-L',RUNTIME/'usr/share/postgresql/16',
                     '--auth=trust','--no-locale','-E','UTF8'])
            command([BIN/'pg_ctl','-D',data,'-l',temporary/'postgres.log',
                     '-o',f"-k {temporary} -p 55448 -c listen_addresses=''",'-w','start'])
            started = True
            command([BIN/'createdb','-h',temporary,'-p','55448','quant_platform_v8'])
            env['QP_PG_DSN'] = f'dbname=quant_platform_v8 host={temporary} port=55448 user=sunhaiwei'
            args = [sys.executable,'-m','pytest','quant_platform/tests/test_postgres_live.py',
                    'quant_platform/tests/test_v5_durable_job_fencing.py',
                    '-q','--junitxml='+str(OUT/'tests.xml')]
            with (OUT/'tests.log').open('w') as tests:
                result = subprocess.run(args,cwd=ROOT,env=env,stdout=tests,stderr=subprocess.STDOUT,timeout=180)
        finally:
            if started:
                command([BIN/'pg_ctl','-D',data,'-m','fast','-w','stop'])
    payload = dict(returncode=None if result is None else result.returncode,
        temporary_cluster=str(temporary), server_stopped=True, tcp_listener=False,
        driver='psycopg2-binary 2.9.10 isolated target', production_database_accessed=False)
    (OUT/'execution.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload))
    if result is None or result.returncode: raise RuntimeError('PostgreSQL regression failed')

if __name__ == '__main__':
    main()
