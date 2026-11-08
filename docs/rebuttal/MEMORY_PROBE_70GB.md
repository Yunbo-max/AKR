# AKR Train-50 rebuttal: 70 GB batch / memory table (H100 80 GB)

Status: **complete** (2026-11-08). Raw data:
`results/probe_mem_70gb_20261108.json`; per-lane receipts in
`results/probe_ranks/`; status trail in `probe_mem_20261108.log`.

## Why this probe exists

The rebuttal cases run on the qz 8xH100 cluster. A previous Qwen run crashed
with NCCL Error 2 while saving a checkpoint in the 76.9 GB regime, so the job
budget must keep `torch.cuda.max_memory_allocated()` at or below about 70 GB
per GPU. The probe finds, per case / model / condition / dataset, the largest
batch whose measured peak stays `<= 70e9` bytes (decimal GB; the JSON also
records a 70 GiB variant). It never saves a checkpoint and never runs an
evaluation: inference workloads run **two forward steps**, C13 LoRA runs **two
real optimizer steps**.

## Protocol

* Entry point: `scripts/probe_mem_70gb.py`; job shells:
  `scripts/qz_akr_memprobe_{setup_env,env,run,fix}_20261108.sh`.
* Ascending batch ladder; the first OOM / over-target batch is only a bracket,
  and a bisection between the last good batch and that bracket determines the
  exact maximum. A batch above the target is never selected.
* Each record contains `max_memory_allocated_bytes`,
  `max_memory_reserved_bytes`, the model base allocation, input token count and
  wall time. Peak statistics are reset and the allocator cache emptied before
  every batch.
* Inference uses `use_cache=False, output_hidden_states=True` (the exact
  `OmniBackend.feature` path). C13 uses the pinned LoRA configuration (rank 8,
  alpha 16, q_proj/v_proj, gradient checkpointing) with a real AdamW step.
* Input policy: per dataset, the N examples closest to the dataset **p95
  duration** (long-bucket batches), not the pathological single file. A separate
  `maxlen_b1` record tests the longest waveform in the dataset at batch 1.
* `lp:1000` is applied with the repository's `prepare_audio` (10th-order
  Butterworth low-pass + 16 kHz resample). Files whose source Nyquist is at or
  below 1 kHz are excluded from selection (see the watkins caveat).
* Qwen audio is capped by the processor at 300 s (`chunk_length=300`,
  `nb_max_frames=30000`); longer files are truncated in the mel/token domain.
  The LLM therefore sees at most 7500 audio tokens per file (25 tokens/s).
* C04 uses the deterministic stratified 50-support draw of
  `train50_protocol.allocate_train_support` (recomputed without canonical-PCM
  hashes; groups are `source_filename`).
* GLM speech-code extraction costs ~12 s per 30 s file on CPU, so the GLM sweep
  encodes at most 8 distinct files per job and repeats them to the requested
  batch; all repeats have the same duration class, so the memory measurement is
  unchanged (`distinct_encodings` is recorded).

## Result table

`Batch@70GB` is the case-level recommendation (the minimum over the datasets
listed in that case's matrix); the last four columns are the per-dataset
maxima measured at that dataset's p95 duration. `Peak GiB` is the limiting
dataset's peak at its recommended batch. `0 (OOM)` means even batch 1 does not
fit. `*` marks values that reached the probe ladder cap (true maximum is
larger). `—` means the dataset is not part of that case's matrix.

| Case | Model | Cond | Op | Batch@70GB | Peak GiB | Profile | dogs | watkins | bats | humbugdb |
|---|---|---|---|---|---|---|---|---|---|---|
| C01 | qwen7b | full | forward | 6 | 62.58 | 80gb | 8 | 9 | 88 | 6 |
| C01 | qwen7b | lp:1000 | forward | 6 | 62.58 | 80gb | 8 | 10 | 88 | 6 |
| C01 | qwen3b | full | forward | 6 | 54.68 | 80gb | 9 | 10 | 95 | 6 |
| C01 | qwen3b | lp:1000 | forward | 6 | 54.68 | 80gb | 9 | 10 | 95 | 6 |
| C02 | qwen7b | full | forward | 6 | 62.58 | 80gb | 8 | 9 | 88 | 6 |
| C02 | qwen7b | lp:1000 | forward | 6 | 62.58 | 80gb | 8 | 10 | 88 | 6 |
| C03 | qwen7b | full | forward | 6 | 62.58 | 80gb | 8 | 9 | 88 | 6 |
| C03 | qwen7b | lp:1000 | forward | 6 | 62.58 | 80gb | 8 | 10 | 88 | 6 |
| C03 | beats | full | beats | 47 | 65.08 | n/a | 57 | 61 | 256* | 47 |
| C03 | beats | lp:1000 | beats | 47 | 65.08 | n/a | 57 | 50 | 256* | 47 |
| C04 | qwen7b | full | icl50 | 1 | 35.42 | 48gb | 0 (OOM) | 1 | 3 | 0 (OOM) |
| C04 | qwen7b | lp:1000 | icl50 | 1 | 31.92 | 48gb | 0 (OOM) | 1 | 3 | 0 (OOM) |
| C05 | qwen7b | full | forward | 8 | 58.20 | 80gb | 8 | 9 | — | — |
| C05 | qwen7b | lp:1000 | forward | 8 | 58.20 | 80gb | 8 | 10 | — | — |
| C06 | qwen7b | full | forward | 8 | 58.20 | 80gb | 8 | 9 | — | — |
| C06 | qwen7b | lp:1000 | forward | 8 | 58.20 | 80gb | 8 | 10 | — | — |
| C07 | qwen7b | full | forward | 8 | 58.20 | 80gb | 8 | 9 | — | — |
| C07 | qwen7b | lp:1000 | forward | 8 | 58.20 | 80gb | 8 | 10 | — | — |
| C08 | qwen7b | full | forward | 8 | 58.20 | 80gb | 8 | 9 | — | — |
| C08 | qwen7b | lp:1000 | forward | 8 | 58.20 | 80gb | 8 | 10 | — | — |
| C09 | qwen7b | full | forward | 6 | 62.58 | 80gb | 8 | 9 | 88 | 6 |
| C09 | qwen7b | lp:1000 | forward | 6 | 62.58 | 80gb | 8 | 10 | 88 | 6 |
| C10 | qwen7b | full | forward | 6 | 62.58 | 80gb | 8 | 9 | 88 | 6 |
| C10 | qwen7b | lp:1000 | forward | 6 | 62.58 | 80gb | 8 | 10 | 88 | 6 |
| C11 | qwen7b | full | forward | 6 | 62.58 | 80gb | 8 | 9 | 88 | 6 |
| C11 | qwen7b | lp:1000 | forward | 6 | 62.58 | 80gb | 8 | 10 | 88 | 6 |
| C12 | glm4voice9b | full | glm | 1 | 17.87 | 48gb | 1 | 1 | 1 | 1 |
| C12 | glm4voice9b | lp:1000 | glm | 1 | 17.87 | 48gb | 1 | 1 | 1 | 1 |
| C13 | qwen7b | full | lora | 6 | 62.64 | 80gb | 8 | 10 | 88 | 6 |
| C13 | qwen7b | lp:1000 | lora | 6 | 62.64 | 80gb | 8 | 10 | 88 | 6 |
| C14 | qwen7b | full | forward | 8 | 58.20 | 80gb | 8 | 9 | — | — |
| C14 | qwen7b | lp:1000 | forward | 8 | 58.20 | 80gb | 8 | 10 | — | — |
| C14 | qwen3b | full | forward | 9 | 60.83 | 80gb | 9 | 10 | — | — |
| C14 | qwen3b | lp:1000 | forward | 9 | 60.83 | 80gb | 9 | 10 | — | — |
| C15 | qwen7b | full | forward | 6 | 62.58 | 80gb | 8 | 9 | 88 | 6 |
| C15 | naturelm | full | naturelm | 256 | 28.93 | n/a | 256* | 256* | 256* | 256* |

### Longest-file feasibility at batch 1 (`maxlen_b1`)

| Engine | dogs (178.7 s) | watkins (1260 s) | bats (21.9 s) | humbugdb (1794.7 s) |
|---|---|---|---|---|
| qwen7b | ok, 32.9 GB | ok, 59.7 GB | ok, 18.3 GB | ok, 59.7 GB |
| qwen3b | ok, 24.4 GB | ok, 51.2 GB | ok, 9.8 GB | ok, 51.2 GB |
| qwen7b LoRA | ok, 33.0 GB | ok, 59.8 GB | ok, 19.5 GB | ok, 59.8 GB |
| glm4voice9b | ok, 20.2 GB | context-limit (8192) | ok, 19.3 GB | context-limit (8192) |
| beats | ok, 32.6 GB | **OOM at b=1** | ok, 17.3 GB | **OOM at b=1** |
| naturelm | n/a (fixed 10 s windows) | n/a | n/a | n/a |

Qwen's long files fit because the processor truncates them to 300 s. The GLM
context limit is 8192 speech tokens. BEATs on full-length watkins/humbug audio
(1260 s / 1795 s) cannot run even at batch 1.

## Findings that affect the rebuttal runs

1. **Qwen audio batching is quadratic in total audio frames.**
   transformers 4.52.3's audio tower (SDPA path) builds a dense block-diagonal
   `[1, N, N]` mask and materialises per-head attention over the flattened
   batch, where `N = batch × frames_per_audio`. That, not the LLM, sets the
   batch: e.g. qwen7b on 38 s Dogs audio is 29 GB at b=4, 62.5 GB at b=8,
   73.8 GB at b=9 and OOM at b≥10. qwen3b shows the same curve. Switching the
   audio tower to `flash_attention_2` (cu_seqlens path) would change these
   numbers substantially; the probe deliberately mirrors the current runner
   (`attn_implementation='sdpa'`).
2. **C04 (50-audio ICL) was infeasible on Dogs and HumBugDB under SDPA; fixed
   2026-11-08.** The original SDPA result: a 50-support Dogs context (~20.4 k
   tokens) OOMs at b=1; HumBugDB (~25.6 k tokens) OOMs at b=1. Watkin (b=1,
   35.4 GiB) and Bats (b=3) were feasible. This was a property of the SDPA
   audio tower, not of the context budget. The new backend switch
   `icl_audio_attention=flash_attention_2` (C04 config; per-audio `chunked`
   fallback) uses the same `cu_seqlens` blocks with no dense mask or per-head
   materialisation. Re-measured on qz job `job-b4ce760a` (1×H100, 2 steps,
   real `OmniBackend.predict(support=registered_50)`): Dogs full/lp
   **25.813e9** at 19,668 tokens, HumBugDB full/lp **27.965e9** at 24,252
   tokens, both ≤76e9 at batch 1. See
   `results/c04_c16_fix_20261108.json` and §2.3 of
   `BATCH_FLOOR60_20261108.md`. The 50-audio budget is unchanged; the fix is
   attention-implementation churn, not context truncation.
3. **Watkins `lp:1000` native-bandwidth policy (resolved 2026-11-08, commit `1fb3385`).** 67 of 1695 watkins
   recordings have sample rates of 600/640/1280 Hz, so a 1 kHz low-pass is
   physically impossible (their bandwidth is already within the passband).
   Per the PI-approved protocol these recordings are **used as-is** under
   `lp:1000` (identity/pass-through) and recorded in the v2 annotation ledger
   (`INPUT_DOMAIN_EXCLUSIONS.json`, `kind=native_bw_within_passband`).
   All 339 test queries are scored; the registered 50-support (including
   `watkins_train_289`/`watkins_train_914`) participates in fit as-is.
   REPORT fields: `native_bw_queries` (13), `native_bw_query_ids`,
   `native_bandwidth_annotations`, `registered_low_sr_events` (67),
   `scored_queries` (339). Sidecar `*.native_bw.json` marks each such event;
   journals carry `native_bandwidth.used_as_is=true`.
   Note: for these 13 test queries the lp and full conditions are identical —
   disclosed in the reply text; a sensitivity row excluding them is planned.
4. **C12 GLM has two blockers.**
   (a) The current GLM AKR adapter is **batch-one only**
   (`FusedKVBridge` raises `GLM AKR requires batch one and the declared axis
   layout`); the maximum usable batch for C12 is therefore 1, at 19.2-19.5 GB
   decimal (17.9-18.2 GiB) depending on the dataset.
   (b) The pinned `zai-org/glm-4-voice-9b` tokenizer declares only
   `<|audio_0|>`…`<|audio_16382|>` (16383 entries) while the audio quantizer has
   `quantize_vocab_size=16384`; `GLMVoiceBackend.__init__` refuses to build.
   The probe maps the missing `<|audio_16383|>` to the contiguous id 168736
   (inside the 168960-row embedding) and records
   `engine_note=missing_audio_16383_mapped_to_contiguous_id`. The runner needs
   the same fallback or an asset fix before C12 can execute.
5. **C03 BEATs cannot process the longest files.** 1260 s watkins and 1795 s
   humbugdb OOM at batch 1 (the encoder's attention is quadratic in frames).
   BEATs on the p95-level files is fine (Dogs 57, Watkins 50-61, HumBugDB 47;
   Bats reaches the 256 ladder cap at only 3.4 GB).
6. **C15 NatureLM reaches the 256 ladder cap** at 31-55 GB (Watkins 55.4 GB at
   256), so its true maximum is above 256; the table reports `256*`.
7. **Target-unit note.** The primary target is 70e9 bytes (65.2 GiB). If 70 GiB
   is intended, several rows can step one batch higher; the JSON carries
   `recommended_batch_70gib` (e.g. qwen7b Dogs 8 -> 9, batches 47 -> 48 for
   BEATs).
8. **Bracket disclosure.** To bracket the threshold, bisection candidates were
   executed above 70 GB; the largest observed forward/train-step peak was
   81.9 GB (76.3 GiB) for a short 2-step probe. No checkpoint was saved and no
   NCCL collective was involved, but this did enter the 76.9 GB tier during
   measurement. Every **selected** batch is `<= 70 GB`.

## How to use this table

* `--profile` is the runner's **model-loading budget**, not a reservation. The
  table's `Profile` column is the smallest budget that holds the recommended
  batched peak. For actual runner commands, use **`--profile 80gb`** for every
  GPU case that includes watkins/humbugdb: a single 1260 s watkins file at
  batch 1 peaks at 59.7 GB (55.6 GiB), above the 48gb budget (43 GiB). Batches
  under ~43 GiB (C04 watkins/bats, C12) also fit `48gb`.
* The current runner processes one query at a time (batch 1). The batch numbers
  are the headroom for a batched implementation, e.g. batching query feature
  extraction or the LoRA micro-batch. Every recommendation is derived from the
  same code path the runner uses.
* Generation (`predict`) additionally allocates the KV cache (~0.4 GB per 7B
  example at 7.5 k tokens); `support_gradients` additionally runs a backward
  pass. Both need their own (smaller) probe if they will be batched.

### Command templates

```bash
# Standard Qwen case (replace case script, dataset, model, condition).
python experiments/rebuttal_cases/c01_main.py run --execute \
  --dataset dogs --model qwen7b --condition full --run-id final_v1 --profile 80gb

# C03 additionally needs the BEATs/openSMILE assets JSON; mfcc/eGeMAPS are CPU-only.
python experiments/rebuttal_cases/c03_acoustic_features.py run --execute \
  --dataset dogs --model qwen7b --condition full --run-id final_v1 \
  --profile 80gb --assets /absolute/path/to/assets.json

# C12 uses the isolated GLM interpreter (transformers 4.44.1) and batch 1.
/inspire/hdd/global_user/zhangjiaquan-253108540222/.venvs/akr-glm/bin/python \
  experiments/rebuttal_cases/c12_glm_architecture.py run --execute \
  --dataset dogs --model glm4voice9b --condition full --run-id final_v1 --profile 48gb

# C13 (LoRA) uses the same Qwen entry point; the probe's 6 = microbatch ceiling
# for p95-length audio, 8-10 on shorter buckets, 88 on Bats.
python experiments/rebuttal_cases/c13_matched_lora.py run --execute \
  --dataset dogs --model qwen7b --condition full --run-id final_v1 --profile 80gb

# C00 is a CPU-only data audit; no GPU allocation is needed.
python experiments/rebuttal_cases/c00_protocol.py check \
  --dataset dogs --model qwen7b --condition full --run-id final_v1

# C04: run only the feasible datasets (watkins, bats). Dogs/HumBugDB OOM at b=1.
python experiments/rebuttal_cases/c04_audio_icl.py run --execute \
  --dataset watkins --model qwen7b --condition full --run-id final_v1 --profile 80gb

# C16 is BLOCKED: the original RUN.json/EPISODE.json/LOCK.json registration is
# not present locally, so no GPU workload can be defined.
```

## Environment

* Image `docker.sii.shaipower.online/inspire-studio/relay2:v2` (Python 3.12.3,
  torch 2.8.0a0+5228986c39.nv25.06, H100 80 GB).
* `scripts/qz_akr_memprobe_setup_env.sh` builds two read-only overlays on the
  image torch, fully offline from `wheels/`:
  `/inspire/hdd/global_user/zhangjiaquan-253108540222/.venvs/akr-core`
  (transformers 4.52.3, tokenizers 0.21.4, qwen-omni-utils 0.0.9, peft 0.17.1,
  pydantic-settings, resampy, google-cloud-storage) and
  `/inspire/hdd/global_user/zhangjiaquan-253108540222/.venvs/akr-glm`
  (transformers 4.44.1, tokenizers 0.19.1, tiktoken 0.8.0).
* The image has no torchaudio; official 2.8.0 does not match the NVIDIA torch
  ABI. The probe uses torchaudio 2.7.1 and temporarily reports CUDA 12.6 during
  the import (only torchaudio's conservative minor-version check trips; the
  extension loads and only the CPU Kaldi fbank / functional resample are used).
* Reproduce: `bash scripts/qz_akr_memprobe_env_20261108.sh` (env + smoke),
  `bash scripts/qz_akr_memprobe_run_20261108.sh` (8-lane sweep),
  `bash scripts/qz_akr_memprobe_fix_20261108.sh` (GLM + watkins-lp follow-up
  and final merge).

## Artifacts

* `results/probe_mem_70gb_20261108.json` — final merged data (1050
  measurements, 36 case rows).
* `results/probe_mem_70gb_20261108.rank*.json` — per-lane raw receipts.
* `results/probe_mem_70gb_file_durations_20261108.json` — 21,683 waveform
  durations used for the p95 buckets.
* `probe_mem_20261108.log` — every measurement/status line.
* qz jobs: `job-ea73e1be-7235-463d-97fa-0e269a472284` (env/smoke),
  `job-e10b147b-08fa-4fae-88e3-f3b08300fe75` (main sweep),
  `job-57f687a7-b881-4f05-892c-0678640c280e` (follow-up).

## 合并后实测验证（2026-11-08）

Status: **complete**. 在合并修复 commit `8a3af35`（GLM 词表回退 + lp 输入域排除；HEAD `6be6f9b`）后的代码上，用真实 train50 登记数据复跑 36 行选定配置。每行 2 步前向（C13 为 2 步 LoRA optimizer step），不保存 checkpoint、不跑任何正式 test 评测。峰值由 `torch.cuda.max_memory_allocated` 记录，目标为 70e9 bytes（十进制 70 GB = 65.2 GiB）。36/36 行 PASS，1 个附加代表性格子 PASS。

执行 qz 作业：8×H100 作业 `job-ad1cfad6` 因唯一 8-GPU 节点被其他会话占用、连续 46 分钟 Unschedulable 后停止；改用单 H100 串行 lane 作业 `job-dd05c34e` 完成（spec `79fe954a`，8 个 lane 顺序执行，GLM lane 仍使用隔离的 transformers 4.44.1 venv）。更早一次 8-GPU 提交 `job-73a2bedc` 因验证脚本缺少 `job["_durations"]` 立即失败并被停止，修复后重提；结果以 `job-dd05c34e` 为准。

| # | Case | Model | Cond | Dataset | Batch | Peak GiB | Peak GB (1e9) | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|
| 1 | C01 | qwen3b | full | humbugdb | 6 | 54.68 | 58.715 | PASS |  |
| 2 | C01 | qwen3b | lp:1000 | humbugdb | 6 | 54.68 | 58.715 | PASS |  |
| 3 | C01 | qwen7b | full | humbugdb | 6 | 62.58 | 67.199 | PASS |  |
| 4 | C01 | qwen7b | lp:1000 | humbugdb | 6 | 62.58 | 67.199 | PASS |  |
| 5 | C02 | qwen7b | full | humbugdb | 6 | 62.58 | 67.199 | PASS |  |
| 6 | C02 | qwen7b | lp:1000 | humbugdb | 6 | 62.58 | 67.199 | PASS |  |
| 7 | C03 | beats | full | humbugdb | 47 | 65.11 | 69.912 | PASS | BEATs at the 70GB edge in the probe (65.08 GiB) |
| 8 | C03 | beats | lp:1000 | humbugdb | 47 | 65.11 | 69.912 | PASS | BEATs at the 70GB edge in the probe (65.08 GiB) |
| 9 | C03 | qwen7b | full | humbugdb | 6 | 62.58 | 67.199 | PASS |  |
| 10 | C03 | qwen7b | lp:1000 | humbugdb | 6 | 62.58 | 67.199 | PASS |  |
| 11 | C04 | qwen7b | full | watkins | 1 | 35.42 | 38.036 | PASS | support=registered_50; probe: dogs/humbugdb OOM even at batch 1; case-level batch from watkins |
| 12 | C04 | qwen7b | lp:1000 | watkins | 1 | 31.92 | 34.270 | PASS | support=probe_filtered_50; probe: dogs/humbugdb OOM even at batch 1; case-level batch from watkins; registered watkins lp support contains 2 low-sr events -> real runner fail-closed; memory measured with probe-filtered support |
| 13 | C05 | qwen7b | full | dogs | 8 | 58.20 | 62.495 | PASS |  |
| 14 | C05 | qwen7b | lp:1000 | dogs | 8 | 58.20 | 62.495 | PASS |  |
| 15 | C06 | qwen7b | full | dogs | 8 | 58.20 | 62.495 | PASS |  |
| 16 | C06 | qwen7b | lp:1000 | dogs | 8 | 58.20 | 62.495 | PASS |  |
| 17 | C07 | qwen7b | full | dogs | 8 | 58.20 | 62.495 | PASS |  |
| 18 | C07 | qwen7b | lp:1000 | dogs | 8 | 58.20 | 62.495 | PASS |  |
| 19 | C08 | qwen7b | full | dogs | 8 | 58.20 | 62.495 | PASS |  |
| 20 | C08 | qwen7b | lp:1000 | dogs | 8 | 58.20 | 62.495 | PASS |  |
| 21 | C09 | qwen7b | full | humbugdb | 6 | 62.58 | 67.199 | PASS |  |
| 22 | C09 | qwen7b | lp:1000 | humbugdb | 6 | 62.58 | 67.199 | PASS |  |
| 23 | C10 | qwen7b | full | humbugdb | 6 | 62.58 | 67.199 | PASS |  |
| 24 | C10 | qwen7b | lp:1000 | humbugdb | 6 | 62.58 | 67.199 | PASS |  |
| 25 | C11 | qwen7b | full | humbugdb | 6 | 62.58 | 67.199 | PASS |  |
| 26 | C11 | qwen7b | lp:1000 | humbugdb | 6 | 62.58 | 67.199 | PASS |  |
| 27 | C12 | glm4voice9b | full | bats | 1 | 17.87 | 19.184 | PASS | engine_note kind=tokenizer_audio_vocab_gap fallback_applied=True missing=['</audio_16383/>'] |
| 28 | C12 | glm4voice9b | lp:1000 | bats | 1 | 17.87 | 19.184 | PASS | engine_note kind=tokenizer_audio_vocab_gap fallback_applied=True missing=['</audio_16383/>'] |
| 29 | C13 | qwen7b | full | humbugdb | 6 | 62.64 | 67.263 | PASS |  |
| 30 | C13 | qwen7b | lp:1000 | humbugdb | 6 | 62.64 | 67.263 | PASS |  |
| 31 | C14 | qwen3b | full | dogs | 9 | 60.83 | 65.314 | PASS |  |
| 32 | C14 | qwen3b | lp:1000 | dogs | 9 | 60.83 | 65.314 | PASS |  |
| 33 | C14 | qwen7b | full | dogs | 8 | 58.20 | 62.495 | PASS |  |
| 34 | C14 | qwen7b | lp:1000 | dogs | 8 | 58.20 | 62.495 | PASS |  |
| 35 | C15 | naturelm | full | bats | 256 | 28.96 | 31.093 | PASS | probe ladder cap 256; true maximum above 256 |
| 36 | C15 | qwen7b | full | humbugdb | 6 | 62.58 | 67.199 | PASS |  |

| # | Case | Model | Cond | Dataset | Batch | Peak GiB | Peak GB (1e9) | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|
| +1 | C04 | qwen7b | full | bats | 1 | 21.74 | 23.341 | PASS | task: representative C04 bats full icl50 batch 1 |

### lp 输入域台账（watkins，端到端）

* 真实登记扫描：1695 行中 67 行源 Nyquist <= 1 kHz（test 13 / train 40 / valid 14），与探测阶段一致。
* C01 watkins lp:1000 台账：15 条 = 13 条 query + 2 条 support。native payload 的 excluded_query_ids=13、scored=326；`report_case` 重算后 excluded_queries=13、scored_queries=326、registered_low_sr_events=67，全部通过。
* 13 条被排除 query 若误入音频准备会抛 `excluded event ... unexpectedly reached audio preparation`；未排除 query 正常生成 16 kHz wav。
* 支持集 fail-closed（预期行为）：注册的 50-support 含 `watkins_train_289`（1.42 s, 1280 Hz）和 `watkins_train_914`（24 s, 640 Hz），support-using 方法在 `payload()` 处抛 `input-domain-infeasible lp support: [...] ... cannot be silently reduced`。C11 full + fit_lp 同样以 2 条 support 条目 fail-closed；`dataset_scan` 仍记录 67。
* 结论：watkins lp 的 query 排除路径已生效且 REPORT 计数正确；但任何使用注册 50-support 的 watkins lp 方法都会 fail-closed，而不是静默缩减 support。正式运行前需从 lp-feasible train 池重新登记 50-support，或取得显式批准。humbugdb/dogs 的注册 50-support 无低 sr 事件（0 条），不受影响。
* C04 lp watkins 的内存测量使用探测阶段的 condition-filtered support（与探测的 31.92 GiB 一致），表中 Note 已标明；这不代表真实注册 support 可运行。

### GLM C12（合并后的词表回退）

* 用探测阶段建的 `akr-glm` venv（transformers 4.44.1）直接构造合并后的 `akr_closing.glm_backend.GLVoiceBackend`，不再使用探测脚本里的 tokenizer monkeypatch。
* `engine_note` 已出现：`kind=tokenizer_audio_vocab_gap`、`fallback_applied=true`、`missing_tokens=['<|audio_16383|>']`、`fallback_ids={'<|audio_16383|>': 168736}`、`audio_offset=152353`；构建不再抛 `non-contiguous`。
* C12 full/lp、bats、batch 1：17.87 GiB PASS（与探测 17.87 GiB 一致）。GLM adapter 仍只验 batch 1。

### C04 / C15 / C03 代表性格子

* C04 watkins full icl50 b=1：使用真实 `prepare_registration` 的 50-support（与探测近似 support 完全一致），35.42 GiB PASS。
* C04 bats full icl50 b=1（附加）：21.74 GiB PASS。Dogs/HumBugDB 在 batch 1 仍 OOM（探测结论，未重复跑）。
* C15 NatureLM bats b=256：28.96 GiB PASS；256 是探测 ladder 上限，真实最大值 >256。
* C03 BEATs humbugdb b=47：65.11 GiB = 69.912e9 bytes PASS，距 70e9 仅 0.13%，是全部格子中最紧的余量，正式运行不要再上调。

### 产物

* `results/batch_verify_20261108.json` — 36 行 + 1 附加格子的峰值、判定与备注。
* `results/batch_verify_ledger_20261108.json` — watkins lp 台账端到端检查。
* `results/batch_verify_ranks/verify_selected_20261108.rank*.json` — 每 lane 原始记录。
* `results/batch_verify_registrations/{watkins,bats}_s20261107.json` — 真实 `prepare_registration` 输出（已用 `validate_registered_inputs` 全量校验）。
* `batch_verify_20261108.log` — 状态流水（含第一次 harness bug 与 8-GPU 不可调度的记录）。
* 脚本：`scripts/verify_selected_batches_20261108.py`、`scripts/verify_selected_batches_1gpu_20261108.py`、`scripts/qz_akr_batchverify_{20261108,1gpu_20261108}.{sh,json}`。
