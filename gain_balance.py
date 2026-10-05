"""Paired, fixed-terminal warm start; deliberately separate from legacy resume.

The cloud wrapper owns authorization, data/source hashes and the resource lease.
This module never allocates a VM and never changes the legacy trainer.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random

import numpy as np
import torch
from CoreAudioML.networks import SimpleRNN
from CoreAudioML.training import LossWrapper, PreEmph

GAINS = (.0625, .25, .5, .75, 1.)
ARCHITECTURE = dict(model='SimpleRNN', input_size=2, skip=1, output_size=1,
                    unit_type='LSTM', num_layers=1, hidden_size=40, bias_fl=True)
SETTINGS = dict(seed=20261004, epochs=100, segments=3579, frame=22050,
                batch=50, warmup=200, update=1000, epsilon=1e-5,
                validation_every=2, lr=1e-4)
LEGACY_HASHES = {
    'CoreAudioML/training.py': '520ae2d0fc1d529f8248df0806a5dd2c1e50c1613d05ed4bec85249b8cf3fc43',
    'CoreAudioML/networks.py': 'f914e246c80f22f82cf0008a8b3334be8820d67677a831d8530f222367c3833e',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_legacy():
    for name, expected in LEGACY_HASHES.items():
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError('Legacy source hash mismatch: ' + name)


def configure(seed=SETTINGS['seed']):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.set_default_dtype(torch.float32)


def environment():
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                numpy=np.__version__, platform=platform.platform(),
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                threads=torch.get_num_threads(),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def rng_state():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), *n[2:]],
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python'])
    n = s['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(s['torch'].cpu())
    if s['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])


def warm_start(path, expected_hash, device='cpu'):
    if os.environ.get('PET_RESUME_DIR'):
        raise ValueError('Legacy/finalized resume is forbidden; use only pinned B weights')
    verify_legacy()
    if sha256(path) != expected_hash:
        raise ValueError('Initial model hash mismatch')
    data = json.loads(Path(path).read_text())
    if set(data) != {'model_data', 'state_dict'} or data['model_data'] != ARCHITECTURE:
        raise ValueError('Wrong or incomplete warm-start architecture/source')
    model = SimpleRNN(**{k: v for k, v in ARCHITECTURE.items() if k != 'model'}).to(device)
    template = model.state_dict()
    if set(data['state_dict']) != set(template):
        raise ValueError('Incomplete or extra warm-start parameters')
    state = {}
    for k, expected in template.items():
        value = torch.tensor(data['state_dict'][k], dtype=torch.float32, device=device)
        if value.shape != expected.shape or not torch.isfinite(value).all():
            raise ValueError('Invalid warm-start parameter: ' + k)
        state[k] = value
    if sum(x.numel() for x in state.values()) != 7081:
        raise ValueError('Wrong warm-start parameter count')
    model.load_state_dict(state, strict=True)
    return model


def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=SETTINGS['lr'], betas=(.9, .999),
                            eps=1e-8, weight_decay=1e-4, amsgrad=False,
                            foreach=False, fused=False)


def schedule(epochs=100, segments=3579, seed=20261004):
    # An explicit CPU device also works when legacy CUDA defaults are in force.
    gen = torch.Generator(device='cpu').manual_seed(seed)
    return torch.stack([torch.randperm(segments, generator=gen, device='cpu')
                        for _ in range(epochs)])


def check_schedule(plan, epochs=100, segments=3579):
    if plan.device.type != 'cpu' or plan.dtype != torch.int64 or plan.shape != (epochs, segments):
        raise ValueError('Wrong schedule shape, dtype or device')
    expected = torch.arange(segments, device='cpu').expand(epochs, -1)
    if not torch.equal(plan.sort(dim=1).values, expected):
        raise ValueError('Every segment must appear exactly once per epoch')


class Objectives:
    """C calls the unchanged LossWrapper. T masks samples AFTER pre-emphasis."""
    def __init__(self, device='cpu', dtype=torch.float32, epsilon=1e-5):
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('Invalid epsilon')
        self.device = torch.device(device)
        self.epsilon = epsilon
        devices = [self.device.index or torch.cuda.current_device()] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices), torch.device(self.device):
            self.legacy = LossWrapper({'DC': .25, 'ESRPre': .75}, [-.85, 1])
            self.filt = PreEmph([-.85, 1]).to(device=self.device, dtype=dtype)
            # Construct the coefficient in the requested dtype, not FP32 then FP64.
            self.filt.conv_filter.weight.data.copy_(
                torch.tensor([[[-.85, 1.]]], device=self.device, dtype=dtype))
        self.filt.requires_grad_(False)

    def grouped(self, p, y, g):
        if p.shape != y.shape or p.ndim != 3 or p.shape[-1] != 1 or min(p.shape) == 0:
            raise ValueError('Expected nonempty matching time,batch,1 arrays')
        if g.shape == p.shape:
            g = g[..., 0]
        if g.shape != p.shape[:2] or not all(torch.isfinite(v).all() for v in (p, y, g)):
            raise ValueError('Bad Gain shape or nonfinite input')
        gs = torch.tensor(GAINS, device=g.device, dtype=g.dtype)
        mask = (g[..., None] == gs)
        if not mask.any(dim=-1).all():
            raise ValueError('Unknown Gain')
        m = mask.to(p.dtype)
        n = m.sum(dim=(0, 1))
        nb = m.sum(dim=0)  # batch, Gain
        present = n > 0
        with torch.device(self.device):
            fp, fy = self.filt(p, y)
        e = (y - p)[..., 0]
        e_pre = (fy - fp)[..., 0]
        y_pre = fy[..., 0]
        energy_pre = (m * y_pre[..., None].square()).sum((0, 1)) / n.clamp_min(1)
        energy_raw = (m * y[..., 0, None].square()).sum((0, 1)) / n.clamp_min(1)
        r = (m * e_pre[..., None].square()).sum((0, 1)) / n.clamp_min(1)
        r = r / (energy_pre + self.epsilon)
        col_mean = (m * e[..., None]).sum(0) / nb.clamp_min(1)
        columns = (nb > 0).sum(0)
        d = col_mean.square().sum(0) / columns.clamp_min(1) / (energy_raw + self.epsilon)
        weights = present.to(p.dtype) / present.sum()
        per_group = .75 * r + .25 * d
        value = (per_group * weights).sum()
        stats = torch.stack((n, columns, present, weights, energy_pre, energy_raw,
                             (energy_pre <= self.epsilon) & present,
                             (energy_raw <= self.epsilon) & present, per_group)).detach()
        return value, stats

    def both(self, p, y, g):
        t, stats = self.grouped(p, y, g)
        # The production C always uses the pinned coefficient/epsilon/order.
        if self.epsilon != 1e-5 or p.dtype != torch.float32:
            raise ValueError('Legacy C requires FP32 and epsilon=1e-5')
        with torch.device(self.device):
            c = self.legacy(p, y)
        return {'C': c, 'T': t}, stats


def train_epoch(model, optim, x, y, order, arm, objectives, bs=50, warmup=200,
                update=1000, stop=None):
    """Original recurrent boundaries, including warmup's first backward path."""
    if arm not in ('C', 'T') or model.hidden is not None:
        raise ValueError('Invalid arm or nonempty initial hidden state')
    if order.numel() != x.shape[1] or not torch.equal(order.cpu().sort().values,
                                                    torch.arange(x.shape[1])):
        raise ValueError('Bad epoch order')
    device = next(model.parameters()).device
    totals = torch.zeros((9, len(GAINS)), device=device)
    losses_total = torch.zeros(2, device=device)
    steps = 0
    model.train()
    for b in range(0, order.numel(), bs):
        ids = order[b:b + bs].to(x.device)
        xb, yb = x[:, ids].to(device), y[:, ids].to(device)
        model(xb[:warmup])
        model.zero_grad()
        for start in range(warmup, xb.shape[0], update):
            if stop is not None:
                stop()  # Failure is terminal; only last complete snapshot is recoverable.
            p = model(xb[start:start + update])
            losses, stats = objectives.both(p, yb[start:start + update],
                                            xb[start:start + update, :, 1])
            loss = losses[arm]
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            if not all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()):
                raise ValueError('Nonfinite gradient')
            optim.step()
            model.detach_hidden()
            model.zero_grad()
            totals += stats
            losses_total += torch.stack([losses['C'].detach(), losses['T'].detach()])
            steps += 1
        model.reset_hidden()
    if not all(torch.isfinite(v).all() for v in model.parameters()):
        raise ValueError('Nonfinite model')
    return dict(steps=steps, mean_losses=dict(zip(('C', 'T'), (losses_total / steps).tolist())),
                group_stat_rows=['samples', 'columns', 'present_updates', 'weight_sum',
                                 'energy_pre_sum', 'energy_raw_sum', 'low_pre_updates',
                                 'low_raw_updates', 'group_loss_sum'],
                groups=list(GAINS), group_stat_sums=totals.tolist())


@torch.no_grad()
def predict(model, x, chunk=22050, stop=None):
    if model.hidden is not None:
        raise ValueError('Prediction requires reset hidden state')
    model.eval()
    out = []
    device = next(model.parameters()).device
    try:
        for start in range(0, len(x), chunk):
            if stop is not None:
                stop()
            out.append(model(x[start:start + chunk].to(device)).cpu())
            model.detach_hidden()
        return torch.cat(out)
    finally:
        model.reset_hidden()


@torch.no_grad()
def validate(model, x, y, objectives, stop=None):
    # Continuous inference, then one full-array objective (as in original val).
    rng = rng_state()
    training = model.training
    try:
        p = predict(model, x, stop=stop)
        device = next(model.parameters()).device
        losses, stats = objectives.both(p.to(device), y.to(device), x[..., 1].to(device))
        return dict(losses={k: float(v) for k, v in losses.items()},
                    group_stats=stats.tolist(), scope='full continuous val, test equals val')
    finally:
        restore_rng(rng)
        model.train(training)


def export_model(model, path):
    if model.hidden is not None:
        raise ValueError('Export only at reset boundary')
    data = dict(model_data=ARCHITECTURE,
                state_dict={k: v.detach().cpu().tolist() for k, v in model.state_dict().items()})
    Path(path).write_text(json.dumps(data, allow_nan=False) + '\n')


class Pair:
    """C(epoch n), T(epoch n), then n+1. Independent Adam and RNG streams."""
    def __init__(self, source, source_hash, plan, contract_hash, device='cpu'):
        check_schedule(plan, *plan.shape)
        self.plan, self.contract_hash = plan.clone(), contract_hash
        self.arms = {}
        for name in ('C', 'T'):
            configure()
            model = warm_start(source, source_hash, device)
            obj = Objectives(device)
            self.arms[name] = dict(model=model, optim=optimizer(model), rng=rng_state(),
                                   epoch=0, steps=0, history=[], objectives=obj)
        self.finalized = False

    def next_arm(self):
        c, t = (self.arms[n]['epoch'] for n in ('C', 'T'))
        if self.finalized or not (c == t or c == t + 1):
            raise ValueError('Finalized or inconsistent paired state')
        if c == t == len(self.plan):
            return None
        return 'C' if c == t else 'T'

    def advance(self, x, y, *, validation=None, stop=None, **geometry):
        name = self.next_arm()
        if name is None:
            raise ValueError('All fixed terminal epochs already completed')
        a = self.arms[name]
        restore_rng(a['rng'])
        result = train_epoch(a['model'], a['optim'], x, y, self.plan[a['epoch']], name,
                             a['objectives'], stop=stop, **geometry)
        a['epoch'] += 1
        a['steps'] += result['steps']
        result['epoch'] = a['epoch']
        if validation is not None and a['epoch'] % 2 == 0:
            result['validation'] = validate(a['model'], *validation, a['objectives'], stop=stop)
        a['history'].append(result)
        a['rng'] = rng_state()
        return name, result

    def save(self, directory, terminal=False):
        directory = Path(directory)
        if terminal and self.next_arm() is not None:
            raise ValueError('Incomplete arms cannot export terminal candidates')
        # Immutable directory: a failed copy never replaces a valid snapshot.
        directory.mkdir(parents=True, exist_ok=False)
        state = dict(schema='gain-balance-pair-v1', contract_hash=self.contract_hash,
                     plan=self.plan, runtime=environment(), finalized=terminal, arms={})
        files = {}
        for name, a in self.arms.items():
            if a['model'].hidden is not None:
                raise ValueError('Checkpoint requires reset hidden state')
            state['arms'][name] = dict(model=a['model'].state_dict(), optimizer=a['optim'].state_dict(),
                                      rng=a['rng'], epoch=a['epoch'], steps=a['steps'], history=a['history'])
            path = directory / (name + ('-terminal.json' if terminal else '-current.json'))
            export_model(a['model'], path)
            files[path.name] = sha256(path)
        torch.save(state, directory / 'pair.pt')
        files['pair.pt'] = sha256(directory / 'pair.pt')
        receipt = dict(schema='gain-balance-pair-v1', contract_hash=self.contract_hash,
                       finalized=terminal, files=files)
        (directory / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
        self.finalized = terminal
        return receipt

    def restore(self, directory):
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        if (manifest.get('schema') != 'gain-balance-pair-v1' or manifest.get('finalized') is not False
                or manifest.get('contract_hash') != self.contract_hash):
            raise ValueError('Wrong contract or legacy/finalized resume forbidden')
        if set(manifest['files']) != {'pair.pt', 'C-current.json', 'T-current.json'}:
            raise ValueError('Incomplete paired checkpoint')
        for name, digest in manifest['files'].items():
            if sha256(directory / name) != digest:
                raise ValueError('Checkpoint hash mismatch')
        s = torch.load(directory / 'pair.pt', map_location='cpu', weights_only=True)
        if (s['schema'] != manifest['schema'] or s['contract_hash'] != self.contract_hash
                or s['finalized'] or s['runtime'] != environment() or not torch.equal(s['plan'], self.plan)):
            raise ValueError('Resume numerical environment, state or plan mismatch')
        for name, a in self.arms.items():
            saved = s['arms'][name]
            if not 0 <= saved['epoch'] <= len(self.plan) or len(saved['history']) != saved['epoch']:
                raise ValueError('Invalid saved epoch/history')
            if saved['steps'] != sum(h['steps'] for h in saved['history']):
                raise ValueError('Invalid saved step count')
            a['model'].load_state_dict(saved['model'], strict=True)
            a['optim'].load_state_dict(saved['optimizer'])
            if not all(torch.isfinite(v).all() for v in a['model'].parameters()):
                raise ValueError('Nonfinite checkpoint parameters')
            for k in ('rng', 'epoch', 'steps', 'history'):
                a[k] = saved[k]
        self.next_arm()
