"""Actual CPU trainer in separate processes: continuous vs relocated resume."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile
import torch
import reference_training as R

SOURCE = Path(__file__).resolve().parents[1]
PROJECT = SOURCE.parents[1]
SPEC = PROJECT/'Data/ReferenceGain/reference_run.json'


class TrainingResume(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_training(self, name, *, resume=None, pause=None, controlled=False, epochs=4, changes=None, fail=False, baseline=False, snapshots=False):
        repo = self.root/name
        repo.mkdir()
        for file in ('dist_model_recnet.py', 'reference_training.py'):
            if baseline:
                (repo/file).write_bytes(subprocess.check_output(['git','-C',str(SOURCE),'show','fb7e1e3650fc4aca2f87e3c3fb46874415b19c5b:'+file]))
            else:
                shutil.copyfile(SOURCE/file, repo/file)
        shutil.copytree(SOURCE/'CoreAudioML', repo/'CoreAudioML', ignore=shutil.ignore_patterns('.git','__pycache__'))
        spec = json.loads(SPEC.read_text())
        cfg = copy.deepcopy(spec['trainer_arguments'])
        cfg.update(device='fixture', file_name='fixture', load_config='fixture', cuda=0,
                   hidden_size=4, epochs=epochs, validation_f=1, validation_p=8,
                   batch_size=1, up_fr=22050, save_location=str(repo/'Results'))
        if changes: cfg.update(changes)
        for subset in ('train','val','test'):
            folder=repo/'Data'/subset;folder.mkdir(parents=True)
            n=66163 if subset=='train' else 22063
            x=(np.sin(np.arange(n,dtype=np.float32)*np.float32(.017))*np.float32(.05))
            wavfile.write(folder/'fixture-input.wav',44100,np.column_stack([x,np.full(n,.75,np.float32)]))
            wavfile.write(folder/'fixture-target.wav',44100,np.tanh(x*np.float32(2))*.4)
        config=repo/'Configs/fixture.json';R.write(config,cfg)
        result=repo/'Results/fixture-fixture'
        env={k:v for k,v in os.environ.items() if not k.startswith(('PET_', 'TRAINER_CHECKPOINT_'))}
        env.update(PYTHONHASHSEED=str(cfg['seed']), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   PET_RUN_ID='resume-test', PET_EXPECTED_CONFIG=str(config), PET_PROVENANCE_DIR=str(result/'provenance'),
                   PET_REFERENCE_SPEC=str(SPEC), OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        if resume: env['PET_RESUME_DIR']=str(resume)
        if pause: env['PET_PAUSE_AFTER_EPOCH']=str(pause)
        if snapshots:
            compat=repo/'compatibility.json'
            R.write(compat,{'config':cfg,'reference_initialization':{'run_id':'resume-test','seed':cfg['seed']}})
            env.update(PYTHONPATH=str(PROJECT/'Colab'),PET_CHECKPOINT_DIR=str(repo/'snapshots'),
                       PET_COMPAT_PATH=str(compat),PET_TRAINER_REPO=str(repo),PET_ATTEMPT_ID=name,
                       TRAINER_CHECKPOINT_CALLBACK='run_contract:checkpoint_boundary')
        # Controlled plateau exercises scheduler reductions and accumulated
        # early stopping, while the actual recurrent training still runs.
        wrapper = '''import runpy, torch
from CoreAudioML.networks import SimpleRNN
original = SimpleRNN.process_data
def plateau(self,*a,**kw):
    out, loss = original(self,*a,**kw)
    return out, torch.tensor(1., device=out.device)
SimpleRNN.process_data = plateau
runpy.run_path('dist_model_recnet.py',run_name='__main__')
'''
        if controlled:
            (repo/'fixture_runner.py').write_text(wrapper)
        p=subprocess.run([sys.executable,'-B','fixture_runner.py' if controlled else 'dist_model_recnet.py','-l','fixture'],
                         cwd=repo,env=env,capture_output=True,text=True,timeout=120)
        if fail:
            self.assertNotEqual(p.returncode,0)
            self.assertNotEqual(p.returncode,75)
            return p
        self.assertEqual(p.returncode,75 if pause else 0,p.stdout[-2500:]+p.stderr[-4000:])
        if not baseline:R.verify_training_state(result)
        return result

    def test_checkpoint_addition_preserves_previous_trainer_numerics(self):
        old=self.run_training('baseline',baseline=True,epochs=2,changes={'hidden_size':40})
        new=self.run_training('candidate',epochs=2,changes={'hidden_size':40})
        for file in ('model.json','model_best.json','best_val_out.wav','test_out_final.wav','test_out_best.wav'):
            self.assertEqual(R.digest(old/file),R.digest(new/file),file)
        a,b=R.read(old/'training_stats.json'),R.read(new/'training_stats.json')
        for key in a:
            if 'time' not in key:self.assertEqual(a[key],b[key],key)

    def state(self, folder):
        return torch.load(folder/R.RESUME_STATE, map_location='cpu', weights_only=True)

    def assert_nested_equal(self, a, b):
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.assert_nested_equal(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_nested_equal(x,y)
        else:self.assertEqual(a,b)

    def assert_result_equal(self, a, b):
        for file in ('model.json','model_best.json','best_val_out.wav','test_out_final.wav','test_out_best.wav'):
            self.assertEqual(R.digest(a/file),R.digest(b/file),file)
        x,y=self.state(a),self.state(b)
        for field in ('model','optimizer','scheduler','patience','rng','training_complete'):
            self.assert_nested_equal(x[field],y[field])
        # Timing is deliberately excluded: process restarts take different time.
        for field in x['track']:
            if 'time' not in field:self.assertEqual(x['track'][field],y['track'][field],field)
        self.assertEqual(R.digest(a/'provenance/initial_model.json'),R.digest(b/'provenance/initial_model.json'))

    def test_real_training_resume_in_new_process_matches_uninterrupted(self):
        whole=self.run_training('whole')
        part=self.run_training('part',pause=2)
        self.assertFalse((part/'test_out_final.wav').exists())
        restored=self.run_training('restored',resume=part)
        self.assert_result_equal(whole,restored)
        self.assertEqual(R.digest(part/'provenance/initialization.json'),R.digest(restored/'provenance/initialization.json'))

    def test_two_restarts_preserve_scheduler_and_early_stop(self):
        whole=self.run_training('whole',controlled=True,epochs=15)
        self.run_training('part',pause=3,controlled=True,epochs=15,snapshots=True)
        def snapshot(name):
            folder=self.root/name/'snapshots'
            result=folder/R.read(folder/'latest.json')['snapshot']
            self.assertFalse((result/'best_val_out.wav').exists())
            return result
        self.run_training('second',resume=snapshot('part'),pause=8,controlled=True,epochs=15,snapshots=True)
        restored=self.run_training('restored',resume=snapshot('second'),controlled=True,epochs=15,snapshots=True)
        self.assert_result_equal(whole,restored)
        state=self.state(restored)
        self.assertLess(state['track']['current_epoch'],15)
        self.assertLess(state['optimizer']['param_groups'][0]['lr'],.005)
        self.assertGreater(state['patience'],8)

    def test_snapshot_before_first_validation_can_resume(self):
        changes={'validation_f':2}
        whole=self.run_training('whole',changes=changes,snapshots=True)
        part=self.run_training('part',pause=1,changes=changes,snapshots=True)
        snapshots=self.root/'part/snapshots'
        snapshot=snapshots/R.read(snapshots/'latest.json')['snapshot']
        self.assertFalse((snapshot/'model_best.json').exists())
        self.assertEqual(R.read(snapshot/'_snapshot.json')['run_id'],'part')
        restored=self.run_training('restored',resume=snapshot,changes=changes,snapshots=True)
        self.assert_result_equal(whole,restored)

    def test_wrong_contract_corruption_and_finalized_are_rejected(self):
        part=self.run_training('part',pause=2)
        wrong=self.run_training('wrong',resume=part,changes={'learn_rate':.004},fail=True)
        self.assertIn('Resume configuration differs',wrong.stderr)
        (part/R.RESUME_STATE).write_bytes((part/R.RESUME_STATE).read_bytes()+b'corrupt')
        with self.assertRaises(R.SpecError):R.verify_training_state(part)
        final=self.run_training('final')
        rejected=self.run_training('completed',resume=final,fail=True)
        self.assertIn('Completed result cannot',rejected.stderr)

    def test_old_weights_only_checkpoint_is_not_a_resume_checkpoint(self):
        with self.assertRaises(FileNotFoundError):R.verify_training_state(self.root)


if __name__=='__main__':unittest.main()
