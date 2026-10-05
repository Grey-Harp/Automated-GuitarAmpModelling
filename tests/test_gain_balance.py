"""Fixed CPU tolerances, independent scalar formula and actual Adam controls."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import gain_balance as G
from CoreAudioML.networks import SimpleRNN
from CoreAudioML.training import LossWrapper


def oracle(p, y, g, epsilon=1e-5):
    p,y=p[...,0],y[...,0]
    fp,fy=p.copy(),y.copy()
    fp[1:]-=.85*p[:-1];fy[1:]-=.85*y[:-1]
    rows=[]
    for gain in sorted(set(g.ravel())):
        ids=[(t,b) for t in range(len(p)) for b in range(p.shape[1]) if g[t,b]==gain]
        r=np.mean([(fy[t,b]-fp[t,b])**2 for t,b in ids])/(np.mean([fy[t,b]**2 for t,b in ids])+epsilon)
        dc=[]
        for b in range(p.shape[1]):
            ts=[t for t in range(len(p)) if g[t,b]==gain]
            if ts:dc.append(np.mean([y[t,b]-p[t,b] for t in ts])**2)
        d=np.mean(dc)/(np.mean([y[t,b]**2 for t,b in ids])+epsilon)
        rows.append(.75*r+.25*d)
    return np.mean(rows)


def equal(test,a,b):
    if isinstance(a,torch.Tensor):test.assertTrue(torch.equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:equal(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal(test,x,y)
    else:test.assertEqual(a,b)


class GainBalance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):G.configure()

    def fixture(self,folder):
        model=SimpleRNN(**{k:v for k,v in G.ARCHITECTURE.items() if k!='model'})
        path=Path(folder)/'B.json';G.export_model(model,path)
        x=torch.randn(235,7,2)*.04
        x[...,1]=torch.tensor([G.GAINS[i%5] for i in range(7)])
        x[216:,0,1]=.75
        return path,G.sha256(path),x,torch.randn(235,7,1)*.06

    def test_formula_and_gradient(self):
        rng=np.random.default_rng(781)
        for kind in ('one','unequal_mixed','silence','tiny','different_amplitudes'):
            with self.subTest(kind=kind):
                y=rng.normal(size=(5,3,1))*.2;p=rng.normal(size=y.shape)*.1
                g=np.full((5,3),.25)
                if kind!='one':g[:,1]=1.;g[2:,0]=.0625
                if kind=='silence':y[:]=0
                if kind=='tiny':y*=1e-7;p*=1e-7
                if kind=='different_amplitudes':y[:,0]*=.001;y[:,2]*=10
                pt=torch.tensor(p,requires_grad=True)
                got,stats=G.Objectives(dtype=torch.float64).grouped(pt,torch.tensor(y),torch.tensor(g))
                self.assertAlmostEqual(float(got.detach()),oracle(p,y,g),delta=1e-10)
                got.backward();grad=np.empty_like(p)
                for i in np.ndindex(p.shape):
                    a,b=p.copy(),p.copy();a[i]+=1e-6;b[i]-=1e-6
                    grad[i]=(oracle(a,y,g)-oracle(b,y,g))/2e-6
                np.testing.assert_allclose(pt.grad.numpy(),grad,rtol=2e-5,atol=2e-7)
                self.assertEqual(int(stats[0].sum()),15)
                self.assertAlmostEqual(float(stats[3].sum()),1.)
                self.assertEqual(float(stats[3,2]),0.)

    def test_one_gain_matches_legacy_value_and_gradient(self):
        for scale in (0.,1e-7,.2,4.):
            y=torch.randn(73,3,1)*scale;p=(torch.randn_like(y)*.1).requires_grad_()
            losses,_=G.Objectives().both(p,y,torch.full((73,3),.25))
            c=torch.autograd.grad(losses['C'],p,retain_graph=True)[0]
            t=torch.autograd.grad(losses['T'],p)[0]
            torch.testing.assert_close(losses['C'],losses['T'],rtol=2e-6,atol=1e-6)
            torch.testing.assert_close(c,t,rtol=3e-6,atol=1e-6)

    def test_bad_inputs(self):
        y=torch.ones(8,2,1);p=y*.9;g=torch.ones(8,2);obj=G.Objectives()
        for name in ('p','y','g'):
            for bad in (float('nan'),float('inf')):
                v=dict(p=p.clone(),y=y.clone(),g=g.clone());v[name].flatten()[0]=bad
                with self.assertRaises(ValueError):obj.grouped(**v)
        with self.assertRaises(ValueError):obj.grouped(p[:0],y[:0],g[:0])
        with self.assertRaises(ValueError):obj.grouped(p,y,g*.3)

    def test_warm_start_strict_predictions_and_fresh_adam(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,digest,x,y=self.fixture(tmp)
            expected=G.predict(G.warm_start(path,digest),x)
            pair=G.Pair(path,digest,G.schedule(2,7),'fixture')
            for a in pair.arms.values():
                self.assertTrue(torch.equal(G.predict(a['model'],x),expected))
                self.assertEqual(a['optim'].state_dict()['state'],{})
                for k,v in dict(lr=1e-4,eps=1e-8,weight_decay=1e-4,betas=(.9,.999),
                                amsgrad=False,foreach=False,fused=False).items():
                    self.assertEqual(a['optim'].param_groups[0][k],v)
            with self.assertRaises(ValueError):G.warm_start(path,'0'*64)
            original=json.loads(path.read_text())
            for kind in ('architecture','missing','nonfinite'):
                d=copy.deepcopy(original)
                if kind=='architecture':d['model_data']['hidden_size']=39
                elif kind=='missing':del d['state_dict']['lin.bias']
                else:d['state_dict']['lin.bias'][0]=float('nan')
                path.write_text(json.dumps(d))
                with self.assertRaises(ValueError):G.warm_start(path,G.sha256(path))
            with patch.dict(os.environ,PET_RESUME_DIR='finalized-original'):
                with self.assertRaisesRegex(ValueError,'resume'):G.warm_start(path,G.sha256(path))

    def test_control_matches_original_adam_and_cc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,digest,x,y=self.fixture(tmp)
            old,new,other=[G.warm_start(path,digest) for _ in range(3)]
            opt=[G.optimizer(m) for m in (old,new,other)]
            for order in G.schedule(3,7):
                with patch('torch.randperm',return_value=order):
                    old.train_epoch(x,y,LossWrapper({'DC':.25,'ESRPre':.75},[-.85,1]),opt[0],3,200,11)
                for i,model in enumerate((new,other),1):
                    r=G.train_epoch(model,opt[i],x,y,order,'C',G.Objectives(),3,200,11)
                    self.assertEqual(r['steps'],12)
                    equal(self,old.state_dict(),model.state_dict())
                    equal(self,opt[0].state_dict(),opt[i].state_dict())
                before=G.rng_state()
                G.validate(other,x[:,:1],y[:,:1],G.Objectives())
                G.export_model(other,Path(tmp)/'diagnostic.json')
                equal(self,before,G.rng_state())

    def test_schedule(self):
        before=G.rng_state();plan=G.schedule();equal(self,before,G.rng_state())
        G.check_schedule(plan);self.assertTrue(torch.equal(plan,G.schedule()))
        bad=plan.clone();bad[1,0]=bad[1,1]
        with self.assertRaises(ValueError):G.check_schedule(bad)
        self.assertEqual(len(range(200,22050,1000))*len(range(0,3579,50))*100,158400)

    def test_pair_recovery_order_and_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,digest,x,y=self.fixture(tmp);plan=G.schedule(2,7)
            full=G.Pair(path,digest,plan,'fixture');resumed=G.Pair(path,digest,plan,'fixture')
            args=dict(bs=3,warmup=200,update=11,validation=(x[:,:1],y[:,:1]))
            self.assertEqual(full.advance(x,y,**args)[0],'C')
            snap=Path(tmp)/'snapshot';full.save(snap);resumed.restore(snap)
            with self.assertRaises(ValueError):G.Pair(path,digest,plan,'other').restore(snap)
            with self.assertRaises(ValueError):full.save(Path(tmp)/'early',terminal=True)
            for name in ('T','C','T'):
                self.assertEqual(full.advance(x,y,**args)[0],name)
                self.assertEqual(resumed.advance(x,y,**args)[0],name)
                for arm in ('C','T'):
                    a,b=full.arms[arm],resumed.arms[arm]
                    equal(self,a['model'].state_dict(),b['model'].state_dict())
                    equal(self,a['optim'].state_dict(),b['optim'].state_dict())
                    equal(self,a['rng'],b['rng']);equal(self,a['history'],b['history'])
            self.assertIsNone(full.next_arm())
            terminal=Path(tmp)/'terminal';full.save(terminal,terminal=True)
            with self.assertRaises(ValueError):resumed.restore(terminal)
            with self.assertRaises(ValueError):full.advance(x,y,**args)
            raw=bytearray((snap/'pair.pt').read_bytes());raw[-1]^=1;(snap/'pair.pt').write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'hash'):resumed.restore(snap)


if __name__=='__main__':unittest.main()
