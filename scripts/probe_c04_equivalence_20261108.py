#!/usr/bin/env python3
"""Refined real-weight equivalence check for the C04 audio-ICL attention fix.

The first probe ran the fp32 audio-tower comparison with cuDNN TF32 enabled
(PyTorch's default for convolutions), which makes different batch shapes round
differently.  This pass disables TF32, reports reference magnitudes and bf16
ULP context, and keeps the whole-vs-per-audio comparison below the 50-audio
whole-mask size.

Sections
  * bf16 flash-varlen vs SDPA on 6 audios, with max/mean absolute difference
    and the bf16 ULP at the reference maximum;
  * fp32 (IEEE, TF32 off) whole vs per-audio on 6 audios;
  * fp32 (IEEE, TF32 off) per-audio grouping 1 vs 4 on the full 51-audio
    registered context (50 support + query).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.probe_c04_c16_fix_20261108 import (grouped_forward, prepared_audio,
                                                registration, status, write_json)
from akr_closing.core import Query
from akr_closing.pipeline import prompt_for


def main():
    import torch
    from akr_closing.backend import OmniBackend, audio_flash_attention_modules, chunked_audio_forward
    document = dict(generated_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
                    torch=torch.__version__)
    backend = OmniBackend('Qwen/Qwen2.5-Omni-7B',
                          'ae9e1690543ffd5c0221dc27f79834d0294cba00',
                          profile='80gb')
    try:
        reg, rows = registration('dogs')
        condition = 'full'
        support = [(Query(i, prepared_audio(rows[i], condition),
                          rows[i].get('recording_id') or i), rows[i]['label'])
                   for i in reg['splits']['50']['support_ids']]
        query_id = reg['query_ids'][0]
        query = Query(query_id, prepared_audio(rows[query_id], condition),
                      rows[query_id].get('recording_id') or query_id)
        inputs = backend._prepare(query, prompt_for(reg['labels']), support)
        features = inputs['input_features']
        mask = inputs['feature_attention_mask']
        lengths = mask.sum(1)
        flat = features.permute(0, 2, 1)[mask.bool()].permute(1, 0)
        tower = backend.thinker.audio_tower
        dtype = next(tower.parameters()).dtype
        aftercnn, _ = tower._get_feat_extract_output_lengths(lengths)
        small = 6
        span = int(lengths[:small].sum())
        document.update(dataset='dogs', condition=condition, n_audios=int(lengths.numel()),
                        input_tokens=int(inputs['input_ids'].shape[1]),
                        n_frames=int(flat.shape[1]), whole_subset_audios=small)
        with torch.inference_mode():
            reference = tower(flat[:, :span], feature_lens=lengths[:small],
                              aftercnn_lens=aftercnn[:small]).last_hidden_state
        document['bf16_reference'] = dict(abs_max=float(reference.abs().max()),
                                          abs_mean=float(reference.abs().mean()))
        modules, version = audio_flash_attention_modules(tower)
        originals = [layer.self_attn for layer in tower.layers]
        try:
            for layer, module in zip(tower.layers, modules):
                layer.self_attn = module
            with torch.inference_mode():
                flash = tower(flat[:, :span], feature_lens=lengths[:small],
                              aftercnn_lens=aftercnn[:small]).last_hidden_state
        finally:
            for layer, module in zip(tower.layers, originals):
                layer.self_attn = module
        diff = (flash - reference).abs()
        ulp = float(torch.finfo(torch.bfloat16).eps * reference.abs().max())
        document['bf16_flash_vs_sdpa'] = dict(
            flash_attn_version=version, max_abs=float(diff.max()), mean_abs=float(diff.mean()),
            reference_abs_max=document['bf16_reference']['abs_max'],
            max_abs_in_bf16_ulps_at_reference_max=float(diff.max()) / ulp if ulp else None,
            bf16_ulp_at_reference_max=ulp)
        status(f'EQUIV bf16 flash_vs_sdpa max_abs={float(diff.max()):.6f} '
               f'mean_abs={float(diff.mean()):.6f} ref_abs_max={document["bf16_reference"]["abs_max"]:.3f}')

        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        tower.float()
        try:
            with torch.inference_mode():
                whole = tower(flat[:, :span].float(), feature_lens=lengths[:small],
                              aftercnn_lens=aftercnn[:small]).last_hidden_state
                per_audio = chunked_audio_forward(tower.forward, flat[:, :span].float(),
                                                  lengths[:small], aftercnn[:small]).last_hidden_state
                one = grouped_forward(tower.forward, flat.float(), lengths, aftercnn, 1)
                four = grouped_forward(tower.forward, flat.float(), lengths, aftercnn, 4)
            d_whole = (whole - per_audio).abs()
            d_fifty = (one - four).abs()
            document['fp32_ieee'] = dict(
                whole_vs_per_audio_max_abs=float(d_whole.max()),
                whole_vs_per_audio_mean_abs=float(d_whole.mean()),
                whole_vs_per_audio_reference_abs_max=float(whole.abs().max()),
                fifty_audio_group1_vs_group4_max_abs=float(d_fifty.max()),
                fifty_audio_group1_vs_group4_mean_abs=float(d_fifty.mean()),
                fifty_audio_reference_abs_max=float(one.abs().max()))
            status(f'EQUIV fp32_ieee whole_vs_per_audio_max_abs={float(d_whole.max()):.3e} '
                   f'fifty_group1_vs_group4_max_abs={float(d_fifty.max()):.3e}')
        finally:
            tower.to(dtype)
    finally:
        handle = getattr(backend, '_count_handle', None)
        if handle is not None:
            handle.remove()
        del backend
    write_json(ROOT / 'results/c04_equivalence_20261108.json', document)
    status(f'EQUIV_DONE {ROOT / "results/c04_equivalence_20261108.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
