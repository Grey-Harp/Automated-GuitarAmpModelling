"""Configuration and optional lifecycle integration without cloud credentials."""
import argparse
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import reference_training as R


class ReferenceTraining(unittest.TestCase):
    def test_configured_zero_seed_is_applied_without_reference_mode(self):
        with patch.dict(os.environ, {}, clear=True):
            R.configure(argparse.Namespace(seed=0))
            a = (random.random(), np.random.rand(), torch.rand(4))
            R.configure(argparse.Namespace(seed=0))
            b = (random.random(), np.random.rand(), torch.rand(4))
            self.assertEqual(a[:2],b[:2]);torch.testing.assert_close(a[2],b[2],rtol=0,atol=0)
            R.configure(argparse.Namespace(seed=1))
            self.assertFalse(torch.equal(a[2],torch.rand(4)))

    def test_partial_reference_configuration_is_rejected(self):
        with patch.dict(os.environ, {'PET_RUN_ID':'incomplete'}, clear=True):
            with self.assertRaises(R.SpecError): R.configure(argparse.Namespace(seed=0))

    def test_effective_argument_mismatch_stops_before_seed_or_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'config.json';R.write(config,{'epochs':2,'seed':0})
            with patch.dict(os.environ,PET_EXPECTED_CONFIG=str(config),PET_PROVENANCE_DIR=tmp,PET_RUN_ID='fixture'),patch.object(R,'apply_seed') as seed:
                R.audit_arguments(argparse.Namespace(epochs=2,seed=0));seed.assert_called_once_with(0)
                with self.assertRaises(R.SpecError): R.audit_arguments(argparse.Namespace(epochs=3,seed=0))
                seed.assert_called_once_with(0)
                self.assertEqual(R.read(Path(tmp)/'effective_config.json')['arguments']['epochs'],2)

    def test_callback_is_optional_and_bad_callback_fails(self):
        with patch.dict(os.environ, {}, clear=True): R.checkpoint('unused')
        with patch.dict(os.environ, TRAINER_CHECKPOINT_CALLBACK='malformed'):
            with self.assertRaises(R.SpecError): R.checkpoint('unused')


if __name__ == '__main__': unittest.main()
