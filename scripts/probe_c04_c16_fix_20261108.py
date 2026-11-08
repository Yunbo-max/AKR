#!/usr/bin/env python3
"""C04 audio-ICL attention fix + C16 batch-1 peak probe (2 steps; no science).

Run inside a qz H100 job via scripts/qz_akr_c04c16_fix_20261108.sh.  It:

* verifies the image's flash-attn package with a tiny varlen call and picks the
  audio-ICL attention mode (flash when usable, otherwise the exact chunked
  per-audio fallback);
* measures C04 dogs / humbugdb 50-audio ICL batch 1 on the *real* OmniBackend
  path used by the runner (2 predict steps per cell, max_new_tokens=32);
* checks the fix on real weights: fp32 per-audio vs grouped forward on all 50
  prepared audios, fp32 whole-vs-per-audio on a small subset, and bf16
  flash-vs-SDPA;
* runs the C16 preflight/plan-produced registration through FinalRunner and
  measures feature / support-gradient / predict batch-1 peaks on two real
  MarmAudio samples.

No checkpoint is saved and no formal experiment is executed.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
LOG = ROOT / 'c04c16_fix_20261108.log'
C16_SOURCE = Path('/tmp/historical_rerun_selfcheck_20261108/source')


def status(message):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {message}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(tmp, path)


def gpu_env():
    import torch
    import transformers
    info = dict(torch=torch.__version__, cuda=torch.version.cuda,
                transformers=transformers.__version__,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                capability=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
                flash_attn=None, flash_varlen_ok=False)
    try:
        import flash_attn
        info['flash_attn'] = flash_attn.__version__
        q = torch.randn(7, 2, 64, device='cuda', dtype=torch.bfloat16)
        cu = torch.tensor([0, 3, 7], dtype=torch.int32, device='cuda')
        out = flash_attn.flash_attn_varlen_func(q, q, q, cu, cu, 4, 7, dropout_p=0.0)
        torch.cuda.synchronize()
        info['flash_varlen_ok'] = tuple(out.shape) == (7, 2, 64)
    except Exception as error:  # noqa: BLE001 - report instead of dying
        info['flash_error'] = repr(error)
    return info


def registration(dataset):
    path = (ROOT / 'results/akr_train50/final_v1/c02'
            / f'{dataset}__qwen7b__full__s20261107' / 'REGISTRATION.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    return data, {row['event_id']: row for row in data['rows']}


def source_sample_rate(path):
    import soundfile as sf
    return int(sf.info(str(path)).samplerate)


def condition_ok(path, condition):
    if not condition.startswith('lp:'):
        return True
    return source_sample_rate(path) / 2.0 > float(condition.split(':', 1)[1])


def prepared_audio(row, condition):
    from akr_closing.core import file_hash
    from akr_closing.pipeline import prepare_audio
    source = Path(row['audio_path'])
    out = (ROOT / 'logs/probe/c04_audio' / condition.replace(':', '_')
           / (file_hash(source)[:20] + '.wav'))
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        prepare_audio(source, out, condition)
    return str(out)


def c04_cell(backend, dataset, condition, reg, rows, *, steps=2, max_new_tokens=32):
    import torch
    from akr_closing.core import Query
    from akr_closing.pipeline import prompt_for
    labels = reg['labels']
    support_ids = reg['splits']['50']['support_ids']
    unsupported = [i for i in support_ids if not condition_ok(rows[i]['audio_path'], condition)]
    if unsupported:
        return dict(status='unsupported_condition', dataset=dataset, condition=condition,
                    event_ids=unsupported)
    support = [(Query(i, prepared_audio(rows[i], condition),
                      rows[i].get('recording_id') or i), rows[i]['label'])
               for i in support_ids]
    query_id = reg['query_ids'][0]
    query = Query(query_id, prepared_audio(rows[query_id], condition),
                  rows[query_id].get('recording_id') or query_id)
    prompt = prompt_for(labels)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step_rows = []
    for step in range(steps):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = backend.predict(query, prompt, labels, support=support,
                                 max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        step_rows.append(dict(
            step=step,
            peak_bytes=int(torch.cuda.max_memory_allocated()),
            peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
            wall_seconds=round(time.perf_counter() - started, 2),
            input_tokens=result['input_tokens'],
            prediction=result['prediction']))
        status(f'C04_STEP {dataset} {condition} step={step} '
               f'peak_gb={step_rows[-1]["peak_bytes"]/1e9:.3f} '
               f'tokens={result["input_tokens"]}')
    return dict(status='ok', dataset=dataset, condition=condition, query_id=query_id,
                support_ids=support_ids, steps=step_rows,
                peak_bytes=max(row['peak_bytes'] for row in step_rows),
                base_allocated_bytes=int(torch.cuda.memory_allocated()))


def grouped_forward(forward, features, lengths, aftercnn, group):
    import torch
    pieces = []
    offset = 0
    for start in range(0, int(lengths.numel()), group):
        stop = min(start + group, int(lengths.numel()))
        span = int(lengths[start:stop].sum())
        piece = forward(features[:, offset:offset + span],
                        feature_lens=lengths[start:stop],
                        aftercnn_lens=aftercnn[start:stop])
        pieces.append(piece.last_hidden_state)
        offset += span
    return torch.cat(pieces, dim=0)


def c04_equivalence(backend, reg, rows, condition):
    """Real-weight checks; 50-audio contexts never use the dense whole mask."""
    import torch
    from akr_closing.backend import audio_flash_attention_modules, chunked_audio_forward
    from akr_closing.core import Query
    from akr_closing.pipeline import prompt_for
    labels = reg['labels']
    support = [(Query(i, prepared_audio(rows[i], condition),
                      rows[i].get('recording_id') or i), rows[i]['label'])
               for i in reg['splits']['50']['support_ids']]
    query_id = reg['query_ids'][0]
    query = Query(query_id, prepared_audio(rows[query_id], condition),
                  rows[query_id].get('recording_id') or query_id)
    inputs = backend._prepare(query, prompt_for(labels), support)
    features = inputs.get('input_features')
    mask = inputs.get('feature_attention_mask')
    if features is None or mask is None:
        return dict(status='no_input_features')
    tower = backend.thinker.audio_tower
    dtype = next(tower.parameters()).dtype
    lengths = mask.sum(1)
    flat = features.permute(0, 2, 1)[mask.bool()].permute(1, 0)
    aftercnn, _ = tower._get_feat_extract_output_lengths(lengths)
    result = dict(status='ok', n_audios=int(lengths.numel()),
                  input_tokens=int(inputs['input_ids'].shape[1]),
                  n_frames=int(flat.shape[1]))
    small = 6
    tower.float()
    try:
        with torch.inference_mode():
            whole = tower(flat[:, :int(lengths[:small].sum())].float(),
                          feature_lens=lengths[:small],
                          aftercnn_lens=aftercnn[:small]).last_hidden_state
            per_audio = chunked_audio_forward(
                tower.forward, flat[:, :int(lengths[:small].sum())].float(),
                lengths[:small], aftercnn[:small]).last_hidden_state
            result['fp32_whole_vs_per_audio_max_abs'] = float((whole - per_audio).abs().max())
            # All fifty audios: two exact block groupings must agree.
            one = grouped_forward(tower.forward, flat.float(), lengths, aftercnn, 1)
            four = grouped_forward(tower.forward, flat.float(), lengths, aftercnn, 4)
            result['fp32_fifty_audio_group1_vs_group4_max_abs'] = float((one - four).abs().max())
        status(f'C04_EQUIVALENCE fp32 whole_vs_chunk={result["fp32_whole_vs_per_audio_max_abs"]:.3e} '
               f'fifty_group1_vs_group4={result["fp32_fifty_audio_group1_vs_group4_max_abs"]:.3e}')
    finally:
        tower.to(dtype)
    if os.environ.get('AKR_FLASH_VARLEN_OK') == '1':
        modules, version = audio_flash_attention_modules(tower)
        originals = [layer.self_attn for layer in tower.layers]
        span = int(lengths[:small].sum())
        with torch.inference_mode():
            sdpa = tower(flat[:, :span], feature_lens=lengths[:small],
                         aftercnn_lens=aftercnn[:small]).last_hidden_state
            try:
                for layer, module in zip(tower.layers, modules):
                    layer.self_attn = module
                flash = tower(flat[:, :span], feature_lens=lengths[:small],
                              aftercnn_lens=aftercnn[:small]).last_hidden_state
            finally:
                for layer, module in zip(tower.layers, originals):
                    layer.self_attn = module
        result['bf16_flash_vs_sdpa_max_abs'] = float((flash - sdpa).abs().max())
        result['bf16_flash_vs_sdpa_mean_abs'] = float((flash - sdpa).abs().mean())
        result['flash_attn_version'] = version
        status(f'C04_EQUIVALENCE bf16 flash_vs_sdpa max_abs={result["bf16_flash_vs_sdpa_max_abs"]:.3e}')
    return result


def measure_c16():
    import torch
    import yaml
    from akr_closing.pipeline import prompt_for
    from akr_final.runner import FinalRunner
    if not (C16_SOURCE / 'C16_OPTIONS.json').is_file():
        return dict(status='missing_preflight_source', source=str(C16_SOURCE))
    options = yaml.safe_load((C16_SOURCE / 'C16_OPTIONS.json').read_text(encoding='utf-8'))
    run_id = 'memprobe_20261108_c16'
    out = ROOT / 'results/akr_historical_readouts' / run_id
    shutil.rmtree(out, ignore_errors=True)  # probe-only output; never a result
    runner = FinalRunner(ROOT, C16_SOURCE, run_id, options, '80gb',
                         output_root=ROOT / 'results/akr_historical_readouts')
    try:
        runner.preflight()
        episode = runner.plan['episodes'][0]
        prompt = prompt_for(runner.labels)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        backend = runner._model('qwen7b', episode)
        base = int(torch.cuda.memory_allocated())
        support_id = episode['support_ids'][0]
        query_id = episode['query_ids'][0]
        support_q = runner._query(support_id, episode['condition'])
        query_q = runner._query(query_id, episode['condition'])
        label = runner.lookup[support_id]['label']
        ops = []

        def measure(name, call):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            call()
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated())
            ops.append(dict(op=name, peak_bytes=peak, peak_gib=round(peak / 2**30, 3),
                            wall_seconds=round(time.perf_counter() - started, 2)))
            status(f'C16_OP {name} peak_gb={peak/1e9:.3f} wall={ops[-1]["wall_seconds"]}s')

        measure('feature_forward', lambda: backend.feature(query_q, prompt))
        measure('support_gradient', lambda: backend.support_gradients(support_q, label, prompt))
        measure('predict', lambda: backend.predict(query_q, prompt, runner.labels))
        peak = max(row['peak_bytes'] for row in ops)
        floor = 30_000_000_000
        pad = 0
        if peak < floor:
            pad = int(math.ceil((floor - peak) / 0.5e9) * 0.5e9)
        return dict(status='ok', episode=episode['id'], condition=episode['condition'],
                    support_id=support_id, query_id=query_id,
                    n_support=len(episode['support_ids']), n_query=len(episode['query_ids']),
                    base_allocated_bytes=base, ops=ops, peak_bytes=peak,
                    floor_bytes=floor, pad_suggestion_bytes=pad,
                    pad_suggestion_gib=round(pad / 2**30, 3),
                    pad_command=(f'python scripts/floor60_pad.py --bytes {pad} '
                                 f'--hold-seconds <job_seconds>' if pad else None))
    finally:
        runner.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/c04_c16_fix_20261108.json')
    parser.add_argument('--flash-mode', choices=['auto', 'flash', 'chunked'], default='auto')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--datasets', default='dogs,humbugdb')
    parser.add_argument('--conditions', default='full,lp:1000')
    parser.add_argument('--skip-c16', action='store_true')
    args = parser.parse_args(argv)

    document = dict(generated_at=time.strftime('%Y-%m-%dT%H:%M:%S'), host=os.uname().nodename,
                    environment=gpu_env(), c04=dict(cells=[], equivalence=[]), c16=None)
    status(f'ENV torch={document["environment"]["torch"]} '
           f'transformers={document["environment"]["transformers"]} '
           f'flash={document["environment"]["flash_attn"]} '
           f'varlen_ok={document["environment"]["flash_varlen_ok"]}')
    os.environ['AKR_FLASH_VARLEN_OK'] = '1' if document['environment']['flash_varlen_ok'] else '0'
    mode = 'flash_attention_2' if document['environment']['flash_varlen_ok'] else 'chunked'
    if args.flash_mode == 'flash':
        if not document['environment']['flash_varlen_ok']:
            raise SystemExit('flash requested but the image flash-attn varlen call failed')
        mode = 'flash_attention_2'
    elif args.flash_mode == 'chunked':
        mode = 'chunked'
    document['c04']['mode'] = mode
    status(f'C04_MODE {mode}')

    import torch
    from akr_closing.backend import OmniBackend
    from akr_final.train50_cases import case_plan
    model = case_plan('c04', 'dogs', 'qwen7b', 'full', 20261107)
    backend = None
    try:
        backend = OmniBackend('Qwen/Qwen2.5-Omni-7B',
                              'ae9e1690543ffd5c0221dc27f79834d0294cba00',
                              profile='80gb', icl_audio_attention=mode)
        document['backend_provenance'] = backend.provenance
        status(f'C04_BACKEND loaded base_gb={torch.cuda.memory_allocated()/1e9:.3f}')
        for dataset in args.datasets.split(','):
            reg, rows = registration(dataset)
            for condition in args.conditions.split(','):
                try:
                    cell = c04_cell(backend, dataset, condition, reg, rows, steps=args.steps)
                except Exception as error:  # noqa: BLE001 - keep other cells measurable
                    cell = dict(status='error', dataset=dataset, condition=condition,
                                error=repr(error), traceback=traceback.format_exc()[-2000:])
                document['c04']['cells'].append(cell)
                status(f'C04_CELL {dataset} {condition} status={cell["status"]} '
                       f'peak_gb={cell.get("peak_bytes", 0)/1e9:.3f}')
            try:
                document['c04']['equivalence'].append(
                    dict(dataset=dataset, condition='full',
                         **c04_equivalence(backend, reg, rows, 'full')))
            except Exception as error:  # noqa: BLE001
                document['c04']['equivalence'].append(
                    dict(dataset=dataset, status='error', error=repr(error)))
                status(f'C04_EQUIVALENCE_ERROR {dataset} {error!r}')
    finally:
        if backend is not None:
            handle = getattr(backend, '_count_handle', None)
            if handle is not None:
                handle.remove()
            del backend
        gc.collect()
        torch.cuda.empty_cache()
    if not args.skip_c16:
        try:
            document['c16'] = measure_c16()
            status(f'C16_DONE peak_gb={document["c16"].get("peak_bytes", 0)/1e9:.3f} '
                   f'pad_bytes={document["c16"].get("pad_suggestion_bytes")}')
        except Exception as error:  # noqa: BLE001
            document['c16'] = dict(status='error', error=repr(error),
                                   traceback=traceback.format_exc()[-2000:])
            status(f'C16_ERROR {error!r}')
    write_json(args.out, document)
    status(f'PROBE_DONE {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
