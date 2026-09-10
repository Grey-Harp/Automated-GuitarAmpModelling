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
    if resuming():
        verify_provenance(os.environ['PET_PROVENANCE_DIR'], os.environ['PET_RUN_ID'], args.seed)
        previous = read(Path(os.environ['PET_PROVENANCE_DIR']) / 'effective_config.json')['arguments']
        if numerical_arguments(previous) != numerical_arguments(actual):
            raise SpecError('Resume configuration differs from original initialization')
        apply_seed(args.seed, record=False)
        return
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


def apply_seed(seed, record=True):
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
    if record:
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
    if reference_enabled() and not resuming(): capture_initial(network)


def setup(args, network, optimiser, scheduler):
    if reference_enabled() and not resuming(): audit_setup(args, network, optimiser, scheduler)


def train_start(args):
    if reference_enabled() and not resuming(): capture_train_start(args)


def checkpoint(save_path):
    callback = os.environ.get('TRAINER_CHECKPOINT_CALLBACK')
    if not callback: return
    module, separator, name = callback.partition(':')
    if not separator or not all(part.isidentifier() for part in module.split('.')) or not name.isidentifier():
        raise SpecError('Checkpoint callback must be module:function')
    getattr(importlib.import_module(module), name)(save_path)


# Continuation is explicit and only at a completed epoch boundary. JSON exports
# remain usable by plugins; this separate file is the training state.
RESUME_STATE = 'training_state.pt'
RESUME_MANIFEST = 'training_state.json'
PAUSED_EXIT_CODE = 75


def resuming():
    return bool(os.environ.get('PET_RESUME_DIR'))


def numerical_arguments(arguments):
    # Locations may differ on another machine. Everything controlling training,
    # including the total epoch limit and seed, must be identical.
    return {k: v for k, v in arguments.items()
            if k not in ('save_location', 'config_location', 'load_config')}


def restore_rng(state):
    import random
    import numpy as np
    import torch
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value
    random.setstate(tuples(state['python']))
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(torch.tensor(state['torch_cpu'], dtype=torch.uint8, device='cpu'))
    if len(state['torch_cuda']) != torch.cuda.device_count():
        raise SpecError('CUDA RNG device count differs')
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all([torch.tensor(v, dtype=torch.uint8, device='cpu') for v in state['torch_cuda']])


def continuation_environment(network):
    import platform
    import numpy as np
    import torch
    device = next(network.parameters()).device
    return {'python': platform.python_version(), 'numpy': str(np.__version__), 'torch': str(torch.__version__),
            'device': str(device), 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads()}


def continuation_contract(args, network):
    source = Path(__file__).resolve().parent
    sources = ('reference_training.py', 'dist_model_recnet.py', 'CoreAudioML/networks.py',
               'CoreAudioML/training.py', 'CoreAudioML/dataset.py', 'CoreAudioML/miscfuncs.py')
    files = [Path('Data') / sub / (args.file_name + '-' + kind + '.wav')
             for sub in ('train', 'val', 'test') for kind in ('input', 'target')]
    result = {'arguments': numerical_arguments(vars(args)),
              'sources': {p: digest(source / p) for p in sources},
              'data': {str(p): digest(p) for p in files},
              'environment': continuation_environment(network),
              'run_id': os.environ.get('PET_RUN_ID')}
    if reference_enabled():
        result['initialization_sha256'] = digest(Path(os.environ['PET_PROVENANCE_DIR']) / 'initialization.json')
    return result


def verify_training_state(folder):
    """Verify bytes before torch deserialization, also usable without torch on a client."""
    folder = Path(folder)
    if (folder/RESUME_MANIFEST).is_symlink():
        raise SpecError('Training manifest must not be a symlink')
    m = read(folder / RESUME_MANIFEST)
    if m.get('schema') != 1 or m.get('boundary') != 'epoch_complete':
        raise SpecError('Unknown training checkpoint schema/boundary')
    required = {RESUME_STATE, 'model.json', 'training_stats.json'}
    if m.get('has_best'): required.add('model_best.json')
    if set(m.get('files', {})) != required:
        raise SpecError('Incomplete training checkpoint manifest')
    if type(m.get('epoch')) is not int or m['epoch'] < 1:
        raise SpecError('Invalid checkpoint epoch')
    for name, expected in m['files'].items():
        if (folder/name).is_symlink() or digest(folder/name) != expected:
            raise SpecError('Training checkpoint changed: ' + name)
    stats = read(folder/'training_stats.json')
    if stats['current_epoch'] != m['epoch'] or len(stats['training_losses']) != m['epoch']:
        raise SpecError('Checkpoint epoch and statistics disagree')
    return m


def save_training_state(folder, network, optimiser, scheduler, track, patience, contract, training_complete):
    import torch
    folder = Path(folder)
    if network.hidden is not None:
        raise SpecError('Only completed epochs with reset recurrent state can be saved')
    state = {'schema': 1, 'model': network.state_dict(), 'optimizer': optimiser.state_dict(),
             'scheduler': scheduler.state_dict(), 'track': dict(track), 'patience': patience,
             'rng': rng_state(), 'contract': contract, 'training_complete': bool(training_complete)}
    # The manifest is published last; a crash leaves a rejected mixed pair, while
    # the previous immutable orchestration snapshot remains intact.
    temporary = folder/(RESUME_STATE+'.tmp')
    torch.save(state, temporary)
    os.replace(temporary, folder/RESUME_STATE)
    names = [RESUME_STATE, 'model.json', 'training_stats.json']
    has_best = (folder/'model_best.json').is_file()
    if has_best: names.append('model_best.json')
    write(folder/RESUME_MANIFEST, {'schema': 1, 'boundary': 'epoch_complete',
          'epoch': track['current_epoch'], 'has_best': has_best,
          'training_complete': bool(training_complete), 'finalized': False,
          'files': {name: digest(folder/name) for name in names}})
    verify_training_state(folder)


def restore_training_state(folder, network, optimiser, scheduler, track, contract):
    import torch
    folder = Path(folder)
    manifest = verify_training_state(folder)
    if manifest.get('finalized'):
        raise SpecError('Completed result cannot start another training session')
    # Only tensor/primitive state from our hash-verified artifact, never arbitrary
    # Python objects from a pickle checkpoint.
    state = torch.load(folder/RESUME_STATE, map_location='cpu', weights_only=True)
    if state.get('schema') != 1 or state.get('contract') != contract:
        raise SpecError('Resume data, arguments, sources, environment or run identity differs')
    if state['track'] != read(folder/'training_stats.json'):
        raise SpecError('Resume state and statistics disagree')
    if state['training_complete'] != manifest['training_complete']:
        raise SpecError('Resume termination state disagrees')
    if type(state['patience']) is not int or state['patience'] < 0:
        raise SpecError('Invalid early stopping state')
    exported = read(folder/'model.json')['state_dict']
    if set(exported) != set(state['model']):
        raise SpecError('Training state and exported model keys disagree')
    for name, tensor in state['model'].items():
        if not torch.isfinite(tensor).all() or not torch.equal(
                tensor, torch.tensor(exported[name], dtype=tensor.dtype, device='cpu')):
            raise SpecError('Training state and exported model weights disagree')
    network.load_state_dict(state['model'], strict=True)
    optimiser.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    track.restore_data(state['track'])
    network.reset_hidden()
    restore_rng(state['rng'])  # Last: initialization and data loading may consume RNG.
    return state['patience'], state['training_complete']


def prepare_resume_result(save_path):
    """Copy into an empty destination, retaining the original initialization."""
    import shutil
    if not resuming(): return
    source, target = Path(os.environ['PET_RESUME_DIR']).resolve(), Path(save_path).resolve()
    verify_training_state(source)
    if source == target or target.exists():
        raise SpecError('Resume requires a new, empty result directory')
    target.mkdir(parents=True)
    for name in verify_training_state(source)['files']:
        shutil.copyfile(source/name, target/name)
    shutil.copyfile(source/RESUME_MANIFEST, target/RESUME_MANIFEST)
    if (source/'provenance').is_dir():
        shutil.copytree(source/'provenance', target/'provenance')
    # A carried best output is useful before a later validation improves it.
    if (source/'best_val_out.wav').is_file():
        shutil.copyfile(source/'best_val_out.wav', target/'best_val_out.wav')
    verify_training_state(target)


def finish_training_state(folder):
    """Mark final tests complete without replacing the saved final model with best."""
    import torch
    folder = Path(folder)
    m = read(folder/RESUME_MANIFEST)
    if not m['training_complete']:
        raise SpecError('Cannot finalize unfinished training')
    state = torch.load(folder/RESUME_STATE, map_location='cpu', weights_only=True)
    state['track'] = read(folder/'training_stats.json')
    temporary = folder/(RESUME_STATE+'.tmp')
    torch.save(state, temporary)
    os.replace(temporary, folder/RESUME_STATE)
    m['finalized'] = True
    m['files'] = {name: digest(folder/name) for name in m['files']}
    write(folder/RESUME_MANIFEST, m)
    verify_training_state(folder)


def pause_requested(epoch, started):
    import time
    stop_epoch = os.environ.get('PET_PAUSE_AFTER_EPOCH')
    seconds = os.environ.get('PET_PAUSE_AFTER_SECONDS')
    for value in (stop_epoch, seconds):
        if value is not None and (not value.isdigit() or int(value) <= 0):
            raise SpecError('Pause limits must be positive integers')
    return ((stop_epoch is not None and epoch >= int(stop_epoch)) or
            (seconds is not None and time.monotonic()-started >= int(seconds)))
