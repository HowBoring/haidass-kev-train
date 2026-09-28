"""Evaluate a selected Haidass artifact with Kev's unmodified official benchmark/scorer.

Requires a local checkout of https://github.com/jaredpalmer/kev and frozen
v7/decision-v7 + v4/transfer-v4 partitions downloaded with the pinned HF revision.
Only the model predictor is replaced; rows, clean filtering, metrics and reports
are produced by kev.benchmark.evaluate_records itself.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import tomllib

import torch

from haidass_kev_train.data.packing import collate, encode_record
from haidass_kev_train.evaluation.run import artifact_sha256, sha256_file
from haidass_kev_train.model.decision import load_artifact
from haidass_kev_train.training.sft import configure_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kev-source', required=True)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--suite-root', default='data/raw/kev-official-eval')
    parser.add_argument('--out', required=True)
    parser.add_argument('--include-locked-test', action='store_true',
                        help='Explicitly evaluate locked test after final candidate promotion')
    args = parser.parse_args()
    source, output = Path(args.kev_source).resolve(), Path(args.out)
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != '90990a5fac2995b9faa3190f7d437e84f2067768':
        raise ValueError('Official scorer revision differs from the published comparison snapshot')
    sys.path.insert(0, str(source))
    from kev.benchmark import evaluate_records
    from kev.suite import load_split, read_manifest
    from kev.metrics import metrics
    calibration_spec = importlib.util.spec_from_file_location('official_calibration', source / 'scripts/calibrate_checkpoint.py')
    calibration_module = importlib.util.module_from_spec(calibration_spec)
    calibration_spec.loader.exec_module(calibration_module)
    output.mkdir(parents=True, exist_ok=False)
    configure_runtime(42)
    # Match kev.predictors.LocalPredictor's FP32 reference evaluation, not the
    # BF16 training forward. The artifact/master weights are unchanged.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    model, tokenizer = load_artifact(args.artifact, trainable=False)
    model.cuda().eval()
    model_hash = artifact_sha256(args.artifact)
    suites = {'decision-v7': Path(args.suite_root) / 'v7/decision-v7',
              'transfer-v4': Path(args.suite_root) / 'v4/transfer-v4'}
    provenance = {'artifact': args.artifact, 'artifact_sha256': model_hash,
                  'official_source_commit': commit,
                  'official_code_sha256': {name: sha256_file(source / name) for name in
                                          ['kev/benchmark.py', 'kev/metrics.py', 'kev/contrastive.py', 'kev/suite.py', 'scripts/calibrate_checkpoint.py']},
                  'suite_repo': 'jaredpalmer/kev-suites',
                  'suite_revision': tomllib.loads(Path('configs/resources.toml').read_text())['data']['kev_suites']['revision'],
                  'inference': {'dtype': 'float32', 'master_weights': 'float32', 'batch_records': 1,
                                'temperature': 1., 'date_facts': False, 'max_packed': 2048},
                  'headline_policy': 'official report.clean: clean knowable questions only; raw probabilities; no test-based selection',
                  'checkpoint_selection': 'selected on development before this evaluation',
                  'locked_test_requested': args.include_locked_test}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

    class Predictor:
        temperature = 1.0

        @torch.inference_mode()
        def __call__(self, record):
            torch.cuda.synchronize()
            started = time.perf_counter()
            encoded = encode_record(record, tokenizer)
            batch = collate([encoded], pad_token_id=tokenizer.pad_token_id or 0).to('cuda')
            hidden = model.backbone(input_ids=batch.input_ids, position_ids=batch.position_ids,
                                    attention_mask=batch.attention_bias.float(), use_cache=False).last_hidden_state
            logits = model.pointer_head(
                hidden[0, batch.decide_positions[0]].unsqueeze(0),
                hidden[0, batch.option_end_positions[0]].unsqueeze(0),
            )[0].masked_fill(~batch.option_mask[0], float('-inf'))
            probabilities = logits.softmax(-1)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            result = {'probabilities': {}, 'logits': {}, 'latency_ms': elapsed, 'inference_temperature': 1.0}
            for index, meta in enumerate(encoded.metadata):
                keys = meta['option_keys']
                result['probabilities'][meta['question_name']] = dict(zip(keys, probabilities[index, :len(keys)].cpu().tolist()))
                result['logits'][meta['question_name']] = dict(zip(keys, logits[index, :len(keys)].cpu().tolist()))
            return result

    predictor = Predictor()
    # Use the official temperature-fit algorithm, but the independent calibration
    # split required by this repository, not the development split used by Kev releases.
    calibration_suite = suites['decision-v7']
    calibration_records = load_split(calibration_suite, 'calibration')
    _, calibration_rows = evaluate_records(calibration_records, predictor, output / 'calibration-raw')
    clean_calibration = [row for row in calibration_rows if row['variant'] == 'clean']
    temperature = calibration_module.fit(clean_calibration)
    calibration = {'model_artifact_sha256': model_hash, 'temperature': temperature,
                   'split': 'decision-v7/calibration', 'split_sha256': sha256_file(calibration_suite / 'calibration.jsonl'),
                   'fit_method': 'official 121-point log grid 0.25..4 minimizing clean micro NLL',
                   'count': len(clean_calibration), 'raw': metrics(clean_calibration),
                   'calibrated': metrics(clean_calibration, temperature)}
    (output / 'calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    summary = {'provenance': provenance, 'calibration': calibration, 'results': {}}
    for split in (('development', 'test') if args.include_locked_test else ('development',)):
        for name, suite in suites.items():
            records = load_split(suite, split, allow_test=split == 'test')
            print(json.dumps({'event': 'evaluating', 'suite': name, 'split': split, 'records': len(records)}), flush=True)
            report, _ = evaluate_records(records, predictor, output / name / split, temperature=temperature,
                                         heldout_sources=tuple(read_manifest(suite)['holdout_sources']))
            report.update(suite=name, split=split, suite_manifest_sha256=sha256_file(suite / 'manifest.json'),
                          split_sha256=sha256_file(suite / f'{split}.jsonl'), date_facts=False,
                          official_source_commit=commit, model_artifact_sha256=model_hash)
            (output / name / split / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            summary['results'][f'{name}/{split}'] = report
            (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps({'event': 'scored', 'suite': name, 'split': split, 'raw': report['clean'],
                              'coverage': report['coverage']}), flush=True)
    if artifact_sha256(args.artifact) != model_hash:
        raise RuntimeError('Selected model artifact changed during evaluation')
    print(json.dumps({'event': 'finished', 'summary': str(output / 'summary.json')}), flush=True)


if __name__ == '__main__':
    main()
