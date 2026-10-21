
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np


def compare_csv(actual,expected,rtol,atol):
    with actual.open(encoding='utf-8',newline='') as stream:observed=list(csv.DictReader(stream))
    with expected.open(encoding='utf-8',newline='') as stream:reference=list(csv.DictReader(stream))
    if len(observed)!=len(reference):raise AssertionError('E_RUN_VALIDATION_015')
    maximum=0.
    for a,b in zip(observed,reference):
        if set(a)!=set(b):raise AssertionError('E_RUN_VALIDATION_018')
        for key in a:
            try:av=float(a[key]);bv=float(b[key])
            except ValueError:
                if a[key]!=b[key]:raise AssertionError('E_RUN_VALIDATION_022')
                continue
            if not np.isfinite(av) or not np.isclose(av,bv,rtol=rtol,atol=atol):
                raise AssertionError('E_RUN_VALIDATION_025')
            maximum=max(maximum,abs(av-bv))
    return maximum


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('validation_output'))
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    case=Path(__file__).resolve().parent/'validation_case'
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    checkpoint=output/'fitted.pt'
    steps=[['train','--data',str(case/'observations.json'),'--config',str(case/'configuration.json'),'--output',str(checkpoint)],
           ['predict','--checkpoint',str(checkpoint),'--queries',str(case/'prediction_queries.json'),'--output',str(output/'predictions.csv'),'--draws-output',str(output/'prediction_draws.npz'),'--draws','16','--seed','2021'],
           ['regional','--checkpoint',str(checkpoint),'--queries',str(case/'regional_queries.json'),'--output',str(output/'regional.csv'),'--draws-output',str(output/'regional_draws.npz'),'--draws','16','--seed','2021']]
    timings={}
    started=time.perf_counter()
    for step in steps:
        tick=time.perf_counter()
        subprocess.run([sys.executable,'-m','aquafieldinr',*step,'--device',args.device],check=True)
        timings[step[0]]=time.perf_counter()-tick
    tolerances=json.loads((case/'tolerances.json').read_text(encoding='utf-8'))
    is_cpu=args.device=='cpu'
    differences={name:compare_csv(output/name,case/f'expected_{name}',tolerances['relative'],tolerances['absolute']) for name in ('predictions.csv','regional.csv')} if is_cpu else None
    if not is_cpu:
        for filename,count in [('predictions.csv',1),('regional.csv',7)]:
            with (output/filename).open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
            if len(rows)!=count:raise AssertionError('E_RUN_VALIDATION_053')
            for row in rows:
                if not all(np.isfinite(float(row[key])) for key in ('mean','sd','q025','q50','q975')):raise AssertionError('E_RUN_VALIDATION_055')
                if not float(row['q025'])<=float(row['q50'])<=float(row['q975']):raise AssertionError('E_RUN_VALIDATION_056')
    for filename,rows in [('prediction_draws.npz',1),('regional_draws.npz',7)]:
        arrays=np.load(output/filename)
        if arrays['draws'].shape!=(rows,16) or not np.isfinite(arrays['draws']).all():raise AssertionError('E_RUN_VALIDATION_059')
        if not np.array_equal(arrays['realization_id'],np.arange(16)):raise AssertionError('E_RUN_VALIDATION_060')
    result={'status':'passed','device':args.device,'verification':'cpu_reference_regression' if is_cpu else 'cuda_execution_and_output_structure','commands':timings,'total_seconds':time.perf_counter()-started,'maximum_absolute_difference':differences,'tolerances':tolerances}
    (output/'validation_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
