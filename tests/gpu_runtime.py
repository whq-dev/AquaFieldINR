
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aquafieldinr.runtime import resolve_device,isolated_rng
from aquafieldinr.training import fit,save,load
from aquafieldinr.dispatch import dispatch
CASE=ROOT/'validation_case'

class RuntimeTests(unittest.TestCase):
    def test_device_and_rng(self):
        self.assertEqual(str(resolve_device('cpu')),'cpu')
        with self.assertRaises(ValueError):resolve_device('mps')
        state=torch.random.get_rng_state().clone()
        with isolated_rng('cpu',19):torch.randn(5)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):resolve_device('cuda')
        else:
            state=torch.cuda.get_rng_state(0).clone()
            with isolated_rng('cuda:0',19):torch.randn(5,device='cuda:0')
            self.assertTrue(torch.equal(state,torch.cuda.get_rng_state(0)))

    def test_failure_cleanup(self):
        from aquafieldinr.runtime import close_posterior
        from aquafieldinr.cli import predict
        config=json.loads((CASE/'configuration.json').read_text());config.update(joint_steps=1,refine_steps=0,starts=1)
        with patch('aquafieldinr.training.close_posterior',wraps=close_posterior) as cleanup:
            with patch('aquafieldinr.training.log_likelihood',side_effect=RuntimeError('injected likelihood failure')):
                with self.assertRaisesRegex(RuntimeError,'injected'):fit(CASE/'observations.json',config)
            self.assertEqual(cleanup.call_count,1)
        checkpoint=fit(CASE/'observations.json',config)
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);save(checkpoint,directory/'fit.pt');(directory/'invalid.json').write_text('{}')
            with patch('aquafieldinr.cli.close_posterior',wraps=close_posterior) as cleanup:
                with self.assertRaises(KeyError):predict(directory/'fit.pt',directory/'invalid.json',directory/'out.csv',draws=2)
                self.assertEqual(cleanup.call_count,1)

    def test_checkpoint_portability(self):
        config=json.loads((CASE/'configuration.json').read_text());config.update(joint_steps=1,refine_steps=0,starts=1)
        state=torch.random.get_rng_state().clone()
        checkpoint=fit(CASE/'observations.json',config)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertTrue(all(t.device.type=='cpu' for t in checkpoint['posterior_state'].values()))
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'fit.pt';save(checkpoint,path)
            before=torch.random.get_rng_state().clone();cpu,posterior,_=load(path,'cpu');posterior.close()
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
            if torch.cuda.is_available():
                gpu,posterior,_=load(path,'cuda:0')
                self.assertEqual(gpu.scales.device.type,'cuda')
                self.assertTrue(all(torch.equal(t.detach().cpu(),checkpoint['model_state'][k]) for k,t in gpu.state_dict().items()))
                posterior.close()
                gpu_checkpoint=fit(CASE/'observations.json',config,device='cuda:0')
                self.assertTrue(all(t.device.type=='cpu' for t in gpu_checkpoint['posterior_state'].values()))
                save(gpu_checkpoint,path);model,posterior,_=load(path,'cpu');posterior.close()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA device is unavailable')
    def test_cuda_dispatch_two_jobs_one_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);config=json.loads((CASE/'configuration.json').read_text());config.update(joint_steps=1,refine_steps=0,starts=1)
            (directory/'fit.json').write_text(json.dumps(config))
            manifest={'jobs':[{'id':f'gpu-fold-{i}','args':['train','--data',str(CASE/'observations.json'),'--config','fit.json','--exclude-position',str(i),'--output','{job_dir}/fit.pt']} for i in range(2)]}
            path=directory/'jobs.json';path.write_text(json.dumps(manifest))
            result=dispatch(path,['cuda:0'],directory/'gpu_jobs')
            self.assertEqual(len(result),2)
            self.assertTrue(all(row['exit_code']==0 and row['device']=='cuda:0' for row in result))
            self.assertEqual([row['id'] for row in result],['gpu-fold-0','gpu-fold-1'])
            for i in range(2):
                model,posterior,cp=load(directory/f'gpu_jobs/gpu-fold-{i}/fit.pt','cpu')
                self.assertIn(str(i),cp['excluded_positions']);posterior.close()

    def test_dispatch_outputs_and_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);config=json.loads((CASE/'configuration.json').read_text());config.update(joint_steps=1,refine_steps=0,starts=1)
            (directory/'fit.json').write_text(json.dumps(config))
            manifest={'jobs':[{'id':f'fold-{i}','args':['train','--data',str(CASE/'observations.json'),'--config','fit.json','--exclude-position',str(i),'--output','{job_dir}/fit.pt']} for i in range(2)]}
            path=directory/'jobs.json';path.write_text(json.dumps(manifest))
            result=dispatch(path,['cpu','cpu'],directory/'successful')
            self.assertEqual(len(result),2);self.assertTrue(all(row['exit_code']==0 for row in result))
            for i in range(2):
                model,posterior,cp=load(directory/f'successful/fold-{i}/fit.pt');self.assertIn(str(i),cp['excluded_positions']);posterior.close()
            with self.assertRaises(FileExistsError):dispatch(path,['cpu'],directory/'successful')
            invalid=copy.deepcopy(manifest);invalid['jobs'][0]['args'][-1]='../escape.pt';path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):dispatch(path,['cpu'],directory/'invalid')
            invalid=copy.deepcopy(manifest);invalid['jobs'][0]['args'][-2:] = ['--output=../escape.pt'];path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):dispatch(path,['cpu'],directory/'invalid_equals')
            manifest['jobs']=manifest['jobs'][:1];manifest['jobs'][0]['args']+=['--invalid-argument'];path.write_text(json.dumps(manifest))
            with self.assertRaises(RuntimeError):dispatch(path,['cpu'],directory/'failed')
            report=json.loads((directory/'failed/dispatch_result.json').read_text());self.assertEqual(report['status'],'failed')

if __name__=='__main__':unittest.main()
