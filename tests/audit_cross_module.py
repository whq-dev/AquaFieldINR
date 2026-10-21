
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
from aquafieldinr.training import fit,save,load,select
CASE=ROOT/'validation_case/observations.json'

class CrossModuleAudit(unittest.TestCase):
    def test_standalone_concentration_selection_removes_optical(self):
        seen=[]
        def scoring(path,config,draws,excluded_positions=(),device="cpu"):
            document=json.loads(Path(path).read_text())
            seen.append(any(r['kind']=='optical' for r in document['records']))
            return [{'position_id':str(i),'variable':'TN','scaled_crps':1.} for i in range(3)]
        with patch('aquafieldinr.training.validate',scoring):
            select(CASE,[{'config':{}}],2,'concentration')
        self.assertTrue(seen)
        self.assertFalse(any(seen),'E_AUDIT_CROSS_MODULE_025')

    def test_corrupt_derived_sparse_structure_is_rejected(self):
        checkpoint=fit(CASE,{'starts':1,'joint_steps':0,'refine_steps':0,'validation_draws':2,'threads':1})
        for name in ('factor_rows','factor_cols','factor_permutation'):
            bad=copy.deepcopy(checkpoint)
            tensor=bad['model_state'][name]
            bad['model_state'][name]=torch.zeros_like(tensor)
            bad['posterior_state']['model.'+name]=bad['model_state'][name].clone()
            with tempfile.TemporaryDirectory() as folder:
                target=Path(folder)/'bad.pt'

                with self.assertRaises(ValueError,msg=name):
                    save(bad,target)
                    _,posterior,_=load(target)
                    posterior.close()

if __name__=='__main__':unittest.main(verbosity=2)
