"""Compare real interrupted and uninterrupted four-update CUDA training runs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import load_file


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def run(output, *args):
    command = [sys.executable, '-m', 'haidass_kev_train.training.sft', '--config',
               'configs/training/pilot.toml', '--output', str(output), *map(str, args)]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        print(result.stdout, result.stderr, flush=True)
        raise RuntimeError(f'Resume verification process failed: {result.returncode}')
    print(json.dumps({'stage': output.name, 'args': list(map(str, args)), 'status': 'passed'}), flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix='kev-resume-') as directory:
        root = Path(directory)
        resumed, control = root / 'resumed', root / 'control'
        run(resumed, '--stop-after', '2')
        run(resumed, '--resume', resumed / 'step-000002')
        run(control)
        left, right = resumed / 'step-000004', control / 'step-000004'
        a, b = load_file(left / 'adapter_model.safetensors'), load_file(right / 'adapter_model.safetensors')
        assert a.keys() == b.keys()
        maxdiff = max(float((a[key] - b[key]).abs().max()) for key in a)
        assert maxdiff == 0, f'Adapter weights diverged: maxdiff={maxdiff}'
        x = torch.load(left / 'training_state.pt', weights_only=False, map_location='cpu')
        y = torch.load(right / 'training_state.pt', weights_only=False, map_location='cpu')
        assert equal(x, y), 'Optimizer/scheduler/RNG/cursor state differs'
        assert json.loads((resumed / 'best.json').read_text()) == json.loads((control / 'best.json').read_text())
        report = {'resume': 'bit_exact', 'adapter_maxdiff': maxdiff, 'global_step': x['global_step'],
                  'epoch': x['epoch'], 'data_cursor': x['data_cursor'], 'optimizer_scheduler_rng_equal': True}
        output = Path('artifacts/reports/sft-resume.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
