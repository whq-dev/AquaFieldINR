import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aquafieldinr.data import Dataset
from aquafieldinr.model import AquaFieldINR
from aquafieldinr.canonical import canonical_representation,canonical_blocks
from aquafieldinr.training import fit,save,load,expand_exclusions,physical_position_groups
from aquafieldinr.workflow import nested_validate
CASE=ROOT/"validation_case/observations.json"

class TrainingAlignment(unittest.TestCase):
    def test_alias_expansion(self):
        doc=json.loads(CASE.read_text())
        first=next(r for r in doc["records"] if r.get("position_id")=="0")
        duplicate=copy.deepcopy(first);duplicate["position_id"]="historical";duplicate["source_position_id"]="source"
        doc["records"].append(duplicate);doc["position_aliases"]={"source":["external"],"external":["extra"]}
        self.assertTrue({"0","historical","source","external","extra"}.issubset(expand_exclusions(doc,["extra"])))

    def test_canonical_invariance(self):
        model=AquaFieldINR(Dataset(CASE),backend="markov")
        with torch.no_grad():
            model.lambda_.fill_(-.3);model.omega.fill_(-.4)
            model.log_tau.copy_(torch.arange(len(model.blocks),0,-1,dtype=torch.float64))
            model.forcing.normal_()
        times=model.data.times;states=torch.randn(model.total_dimension,7)
        view=canonical_representation(model)
        points=np.asarray([[200.,200.],[400.,200.]])

        points=np.asarray(model.data.records[0]["support"])[:1,:2]
        qt=np.repeat(times[0],len(points));ti=np.searchsorted(times,qt)
        bases={ell:model.basis(ell,points) for ell in range(len(model.data.meshes))}
        for kind,variables in (("concentration",("TN","TP","Chla")),("optical",("B2","B4"))):
            for j,variable in enumerate(variables):
                if kind=="optical":j=("B2","B3","B4","B5","B6","B7","B8","B8A").index(variable)
                total=(model.design(qt)@model.background[j])[:,None].expand(-1,7).clone() if kind=="concentration" else model.beta_opt[j].expand(len(points),7).clone()
                for entry,state,concentration,optical,forcing in canonical_blocks(model,states,times,view):
                    z=entry["source_block"];n=model.blocks[z][3]
                    values=state.reshape(len(times),n,-1)[ti]
                    loading=concentration[j] if kind=="concentration" else optical[j]*model.optical_scales[j]
                    total=total+loading*torch.einsum("pn,pns->ps",bases[entry["layer"]],values)
                    self.assertTrue(torch.equal(forcing,model.forcing[z]*entry["sign"]))
                expected=model.field(states,times,points,qt,kind,variable)
                actual=model.scales[j]*torch.nn.functional.softplus(total) if kind=="concentration" else total
                self.assertTrue(torch.allclose(actual,expected,atol=1e-12,rtol=1e-12))

    def test_nested_exclusions_and_stage_inputs(self):
        seen=[]
        def fake_select(path,candidates,draws,stage,excluded,device="cpu"):
            doc=json.loads(Path(candidates[0]["data"]).read_text())
            seen.append((stage,set(excluded),doc))
            self.assertEqual(any(r["kind"]=="optical" for r in doc["records"]),stage=="optical")
            return {"selected_candidate":0,"selected":candidates[0],"evaluations":[]}
        def fake_fit(path,config,excluded,device="cpu"):
            return {"excluded_positions":list(excluded)}
        def fake_score(cp,held,draws,device="cpu"):
            self.assertTrue(all(str(r["position_id"]) in cp["excluded_positions"] for r in held))
            return [{"position_id":r["position_id"],"variable":r["variable"],"scaled_crps":0.1} for r in held]
        with patch("aquafieldinr.workflow.select",fake_select),patch("aquafieldinr.workflow.fit",fake_fit),patch("aquafieldinr.workflow._evaluate",fake_score):
            report,cp=nested_validate(CASE,[{"config":{"shared_rank":1}}],[{"config":{"optical_rank":1,"alpha_gamma":.001}}],draws=2)
        self.assertEqual(len(report["folds"]),3)
        for stage,excluded,doc in seen[:-2]:self.assertEqual(len(excluded),1)
        self.assertEqual(seen[-1][1],set())
        self.assertEqual(len(report["source_sha256"]),1)

    def test_direct_concentration_selection_strips_optical(self):
        from aquafieldinr.training import select
        seen=[]
        original=Dataset(CASE)
        expected_nodes=sum(len(mesh.nodes) for mesh in original.meshes)
        candidate={"data":str(CASE),"config":{"shared_rank":1}}
        def scoring(path,config,draws,excluded,device="cpu"):
            document=json.loads(Path(path).read_text())
            seen.extend(r["kind"] for r in document["records"])
            return [{"position_id":str(i),"variable":"TN","scaled_crps":.1+i*.01} for i in range(3)]
        with patch("aquafieldinr.training.validate",scoring):
            report=select(CASE,[candidate],2,stage="concentration")
        self.assertTrue(seen)
        self.assertEqual(set(seen),{"concentration"})
        self.assertIs(report["selected"],candidate)
        self.assertTrue(Path(report["selected"]["data"]).exists())
        self.assertEqual(report["evaluations"][0]["geometry_nodes"],expected_nodes)

    def test_changed_candidate_rejected(self):
        doc=json.loads(CASE.read_text());doc["records"][0]["value"]*=2
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"changed.json";path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'E_WORKFLOW_070'):
                nested_validate(CASE,[{"data":str(path)}],[{"config":{"optical_rank":1}}],2)

    def test_alias_exclusion_precedes_scaling(self):
        doc=json.loads(CASE.read_text())
        original=next(r for r in doc["records"] if r.get("position_id")=="0")
        alias=copy.deepcopy(original);alias["position_id"]="historical";alias["source_position_id"]="source";alias["value"]=1e6
        doc["records"].append(alias);doc["position_aliases"]={"source":["external"]}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"data.json";path.write_text(json.dumps(doc))
            cp=fit(path,{"starts":1,"joint_steps":0,"refine_steps":0,"validation_draws":2,"threads":1},["external"])
        expected=[]
        for variable in ("TN","TP","Chla"):
            values=[r["value"] for r in doc["records"] if r["kind"]=="concentration" and r["variable"]==variable and r["position_id"] not in {"0","historical"}]
            expected.append(np.median(values)/np.logaddexp(0,1))
        self.assertTrue(np.allclose(cp["model_state"]["scales"],expected))
        self.assertTrue({"0","historical","source","external"}.issubset(cp["excluded_positions"]))

    def test_time_mixture_predictive_score_and_censor(self):
        from aquafieldinr.workflow import _evaluate
        class Model:
            scales=torch.ones(3)
            error_raw=torch.full((3,),-100.)
            def field(self,states,times,points,query_times,kind,variable):
                return torch.tensor([[1.]*1000,[9.]*1000])
        class Posterior:
            closed=False
            def complete(self,*args,**kwargs):return np.array([0.,1.]),None
            def close(self):self.closed=True
        posterior=Posterior()
        held=[{"position_id":"a","variable":"TN","support":[[0,0,0,.5],[0,0,1,.5]],"mode":"instantaneous_time_mixture","value":1.},
              {"position_id":"a","variable":"TP","support":[[0,0,0,1.]],"interval":[0.,.1]}]
        with patch("aquafieldinr.workflow.save"),patch("aquafieldinr.workflow.load",return_value=(Model(),posterior,None)):
            rows=_evaluate({"settings":{"seed":2021}},held,1000)
        self.assertGreater(rows[0]["crps"],1.6)
        self.assertLess(rows[0]["crps"],2.4)
        self.assertEqual(rows[1]["status"],"not_scored")
        self.assertTrue(posterior.closed)

    def test_sparse_train_roundtrip(self):
        cp=fit(CASE,{"starts":1,"joint_steps":1,"refine_steps":1,"draws_per_step":2,"validation_draws":2,"check_every":1,"threads":1})
        self.assertEqual(cp["schema_version"],2)
        self.assertEqual(cp["model_state"]["factor_raw"].ndim,1)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"fit.pt";save(cp,path);model,posterior,restored=load(path)
            self.assertEqual(model.backend,"sparse")
            self.assertEqual(restored["canonical"],cp["canonical"])
            for key,value in cp["model_state"].items():self.assertTrue(torch.equal(value,model.state_dict()[key]))
        for metadata in cp["canonical"]["blocks"]:
            self.assertIn(metadata["sign"],(-1,1))

if __name__=="__main__":unittest.main(verbosity=2)
