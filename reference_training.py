"""Reference training receipts and lifecycle events. No cloud access on import."""
import hashlib
import importlib
import json
import math
import os
from pathlib import Path

class SpecError(ValueError):
    pass

PROVENANCE_FILES = ('effective_config.json', 'training_setup.json', 'seed_receipt.json',
                    'initial_model.json', 'train_start_rng.json')

def load():
    return read(os.environ['PET_REFERENCE_SPEC'])

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def audit_arguments(args):
    expected = read(os.environ['PET_EXPECTED_CONFIG'])
    actual = vars(args)
    if actual != expected:
        keys = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise SpecError('Trainer effective arguments differ: ' + ', '.join(keys))
    apply_seed(args.seed)
    folder = Path(os.environ['PET_PROVENANCE_DIR'])
    write(folder / 'effective_config.json', {'arguments': actual, 'run_id': os.environ['PET_RUN_ID']})


def rng_state():
    import random
    import numpy as np
    import torch
    np_state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [np_state[0], np_state[1].tolist(), *np_state[2:]],
            'torch_cpu': torch.get_rng_state().tolist(),
            'torch_cuda': [state.tolist() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def apply_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise SpecError('Seed must be an explicit uint32 (zero is valid)')
    if os.environ.get('PYTHONHASHSEED') != str(seed):
        raise SpecError('PYTHONHASHSEED must be set before starting the trainer interpreter')
    policy = load()['reproducibility']
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != policy['cublas_workspace_config']:
        raise SpecError('CUBLAS_WORKSPACE_CONFIG must be set before torch is imported')
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(policy['deterministic_algorithms'])
    torch.backends.cudnn.benchmark = policy['cudnn_benchmark']
    torch.backends.cudnn.deterministic = policy['cudnn_deterministic']
    torch.backends.cuda.matmul.allow_tf32 = policy['allow_tf32']
    torch.backends.cudnn.allow_tf32 = policy['allow_tf32']
    receipt = {'seed': seed, 'run_id': os.environ['PET_RUN_ID'],
               'phase': 'after configuration, before model construction',
               'torch': torch.__version__, 'numpy': np.__version__,
               'python_hash_seed': os.environ['PYTHONHASHSEED'],
               'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
               'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
               'cudnn_benchmark': torch.backends.cudnn.benchmark,
               'cudnn_deterministic': torch.backends.cudnn.deterministic,
               'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
               'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
               'rng': rng_state()}
    write(Path(os.environ['PET_PROVENANCE_DIR']) / 'seed_receipt.json', receipt)


def capture_initial(network):
    import torch
    if not all(torch.isfinite(p).all().item() for p in network.parameters()):
        raise SpecError('Initial model contains nonfinite weights')
    folder = Path(os.environ['PET_PROVENANCE_DIR'])
    if (folder / 'initial_model.json').exists():
        raise SpecError('Refusing to overwrite an earlier initialization')
    previous = network.save_state
    try:
        network.save_state = True
        network.save_model('initial_model', str(folder))
    finally:
        network.save_state = previous
    if not read(folder / 'initial_model.json').get('state_dict'):
        raise SpecError('Initial checkpoint omitted weights')


def capture_train_start(args):
    folder = Path(os.environ['PET_PROVENANCE_DIR'])
    if (folder / 'initialization.json').exists():
        raise SpecError('Initialization is already sealed')
    write(folder / 'train_start_rng.json', {'run_id': os.environ['PET_RUN_ID'],
          'phase': 'after loss/optimizer/data construction, before first epoch', 'rng': rng_state()})
    manifest = {'schema': 1, 'run_id': os.environ['PET_RUN_ID'], 'seed': args.seed,
                'files': {name: digest(folder / name) for name in PROVENANCE_FILES}}
    write(folder / 'initialization.json', manifest)
    verify_provenance(folder, manifest['run_id'], args.seed)


def verify_provenance(folder, run_id=None, seed=None):
    folder = Path(folder)
    manifest = read(folder / 'initialization.json')
    if manifest.get('schema') != 1 or set(manifest['files']) != set(PROVENANCE_FILES):
        raise SpecError('Incomplete initialization manifest')
    if run_id is not None and manifest['run_id'] != run_id:
        raise SpecError('Initialization belongs to another run')
    if seed is not None and manifest['seed'] != seed:
        raise SpecError('Initialization seed differs from the run contract')
    for name, expected in manifest['files'].items():
        path = folder / name
        if path.is_symlink() or digest(path) != expected:
            raise SpecError('Initialization artifact changed: ' + name)
    receipt = read(folder / 'seed_receipt.json')
    if receipt['seed'] != manifest['seed'] or receipt['run_id'] != manifest['run_id']:
        raise SpecError('Seed was not applied in this run')
    for name in ('effective_config.json', 'training_setup.json', 'train_start_rng.json'):
        if read(folder / name)['run_id'] != manifest['run_id']:
            raise SpecError('Provenance files belong to different runs')
    if read(folder / 'effective_config.json')['arguments']['seed'] != manifest['seed']:
        raise SpecError('Effective seed differs from applied seed')
    return manifest


def json_state(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    if isinstance(value, dict):
        return {str(k): json_state(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_state(v) for v in value]
    return value


def audit_setup(args, network, optimiser, scheduler):
    spec = load()
    fixed = spec['code_settings']
    groups = [{k: v for k, v in group.items() if k != 'params'} for group in optimiser.param_groups]
    if type(optimiser).__name__ != fixed['optimizer']['name'] or any(
            group['weight_decay'] != fixed['optimizer']['weight_decay'] for group in groups):
        raise SpecError('Unexpected effective optimiser')
    for key, expected in fixed['optimizer']['defaults'].items():
        if json_state(optimiser.defaults.get(key)) != expected:
            raise SpecError('Unexpected effective Adam default: ' + key)
    if optimiser.defaults.get('decoupled_weight_decay', False):
        raise SpecError('Adam weight decay unexpectedly decoupled')
    if any(group['lr'] != args.learn_rate for group in groups):
        raise SpecError('Unexpected initial learning rate')
    for key, expected in fixed['scheduler'].items():
        if key != 'name' and json_state(getattr(scheduler, key)) != expected:
            raise SpecError('Unexpected effective scheduler: ' + key)
    if type(scheduler).__name__ != fixed['scheduler']['name']:
        raise SpecError('Unexpected scheduler class')
    if network.rec.num_layers != fixed['model_layers'] or (network.lin.bias is not None) != fixed['linear_bias']:
        raise SpecError('Unexpected implicit architecture settings')
    import torch
    receipt = {'run_id': os.environ['PET_RUN_ID'], 'torch': torch.__version__,
               'device': str(next(network.parameters()).device),
               'optimizer_defaults': optimiser.defaults, 'optimizer_groups': groups,
               'scheduler_state': scheduler.state_dict(), 'code_settings': fixed,
               'applied_pre_filter': args.pre_filt}
    write(Path(os.environ['PET_PROVENANCE_DIR']) / 'training_setup.json', json_state(receipt))


def reference_enabled():
    keys = ('PET_EXPECTED_CONFIG', 'PET_PROVENANCE_DIR', 'PET_RUN_ID', 'PET_REFERENCE_SPEC')
    present = [bool(os.environ.get(key)) for key in keys]
    if any(present) and not all(present):
        raise SpecError('Reference mode requires all four PET_* identity/configuration paths')
    return all(present)


def configure(args):
    if reference_enabled():
        audit_arguments(args)
    elif args.seed is not None:
        if type(args.seed) is not int or not 0 <= args.seed < 2**32:
            raise SpecError('Seed must be a uint32')
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)


def initial(network):
    if reference_enabled(): capture_initial(network)


def setup(args, network, optimiser, scheduler):
    if reference_enabled(): audit_setup(args, network, optimiser, scheduler)


def train_start(args):
    if reference_enabled(): capture_train_start(args)


def checkpoint(save_path):
    callback = os.environ.get('TRAINER_CHECKPOINT_CALLBACK')
    if not callback: return
    module, separator, name = callback.partition(':')
    if not separator or not all(part.isidentifier() for part in module.split('.')) or not name.isidentifier():
        raise SpecError('Checkpoint callback must be module:function')
    getattr(importlib.import_module(module), name)(save_path)
