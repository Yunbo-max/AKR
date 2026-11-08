#!/usr/bin/env python3
"""Diagnose the residual real-weight equivalence gaps (C04 audio tower).

Controls for the possible explanations of the max-abs outliers:
  * bf16 SDPA whole vs bf16 SDPA per-audio (same kernel, different shapes);
  * bf16 flash-varlen vs bf16 SDPA;
  * fp32 IEEE with cuDNN enabled vs disabled (conv algorithm selection);
  * attention-module level flash vs SDPA on identical hidden states/segments.

No formal experiment; one H100, real Qwen2.5-Omni-7B audio tower.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.probe_c04_c16_fix_20261108 import (prepared_audio, registration,
                                                status, write_json)
from akr_closing.core import Query
from akr_closing.pipeline import prompt_for


def stats(reference, other):
    diff = (other - reference).abs()
    return dict(max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                reference_abs_max=float(reference.abs().max()),
                reference_abs_mean=float(reference.abs().mean()))


def main():
    import torch
    from akr_closing.backend import (OmniBackend, audio_flash_attention_modules,
                                     chunked_audio_forward)
    document = dict(generated_at=time.strftime('%Y-%m-%dT%H:%M:%S'), torch=torch.__version__)
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
        document.update(n_audios=int(lengths.numel()), input_tokens=int(inputs['input_ids'].shape[1]),
                        n_frames=int(flat.shape[1]), subset_audios=small)
        small_flat = flat[:, :span]
        small_lengths = lengths[:small]
        small_after = aftercnn[:small]

        with torch.inference_mode():
            sdpa_whole = tower(small_flat, feature_lens=small_lengths,
                               aftercnn_lens=small_after).last_hidden_state
            sdpa_per = chunked_audio_forward(tower.forward, small_flat, small_lengths,
                                             small_after).last_hidden_state
        document['bf16_sdpa_whole_vs_sdpa_per_audio'] = stats(sdpa_whole, sdpa_per)
        modules, version = audio_flash_attention_modules(tower)
        originals = [layer.self_attn for layer in tower.layers]
        try:
            for layer, module in zip(tower.layers, modules):
                layer.self_attn = module
            with torch.inference_mode():
                flash_whole = tower(small_flat, feature_lens=small_lengths,
                                    aftercnn_lens=small_after).last_hidden_state
        finally:
            for layer, module in zip(tower.layers, originals):
                layer.self_attn = module
        document['bf16_flash_vs_sdpa'] = stats(sdpa_whole, flash_whole)

        hidden = torch.randn(600, tower.config.d_model, device='cuda', dtype=torch.bfloat16)
        cu = torch.tensor([0, 100, 190, 300, 410, 505, 600], dtype=torch.int32, device='cuda')
        with torch.inference_mode():
            attn_sdpa = originals[0](hidden, cu)
            attn_flash = modules[0](hidden, cu)
        document['attention_module_bf16_flash_vs_sdpa'] = stats(attn_sdpa, attn_flash)
        status(f"DIAG attention max_abs={document['attention_module_bf16_flash_vs_sdpa']['max_abs']:.6f} "
               f"mean_abs={document['attention_module_bf16_flash_vs_sdpa']['mean_abs']:.6f}")

        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        tower.float()
        try:
            with torch.inference_mode():
                fp_whole = tower(small_flat.float(), feature_lens=small_lengths,
                                 aftercnn_lens=small_after).last_hidden_state
                fp_per = chunked_audio_forward(tower.forward, small_flat.float(),
                                               small_lengths, small_after).last_hidden_state
            document['fp32_cudnn_on_whole_vs_per_audio'] = stats(fp_whole, fp_per)
            torch.backends.cudnn.enabled = False
            with torch.inference_mode():
                fp_whole_nocudnn = tower(small_flat.float(), feature_lens=small_lengths,
                                         aftercnn_lens=small_after).last_hidden_state
                fp_per_nocudnn = chunked_audio_forward(tower.forward, small_flat.float(),
                                                       small_lengths, small_after).last_hidden_state
            document['fp32_cudnn_off_whole_vs_per_audio'] = stats(fp_whole_nocudnn, fp_per_nocudnn)
            document['fp32_whole_cudnn_on_vs_off'] = stats(fp_whole, fp_whole_nocudnn)
            torch.backends.cudnn.enabled = True
            status(f"DIAG fp32 cudnn_on max={document['fp32_cudnn_on_whole_vs_per_audio']['max_abs']:.3e} "
                   f"cudnn_off max={document['fp32_cudnn_off_whole_vs_per_audio']['max_abs']:.3e} "
                   f"whole on-vs-off max={document['fp32_whole_cudnn_on_vs_off']['max_abs']:.3e}")
        finally:
            tower.to(dtype)
    finally:
        handle = getattr(backend, '_count_handle', None)
        if handle is not None:
            handle.remove()
        del backend
    write_json(ROOT / 'results/c04_equiv_diag_20261108.json', document)
    status(f"DIAG_DONE {ROOT / 'results/c04_equiv_diag_20261108.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
