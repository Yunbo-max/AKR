# NatureLM native specialist comparison (C15)

This adapter is an executable path to the official NatureLM-audio model using
local assets. Its tiny injected-runtime tests validate the adapter contract only.
No official checkpoint was downloaded, loaded, or evaluated during implementation;
GPU readiness, exact checkpoint compatibility, and recognition performance remain
unverified. A missing asset or inference error must be recorded as missing/failed,
not as zero, a generalist prediction, or a successful specialist score.

## Verified primary implementation

Inspected 2026-11-07: the official repository HEAD was
`a4c65ad7d558f902852e20a299eafe927ca0921d` (v1.1.0 version bump).
The adapter requires that exact source revision until a subsequent API is audited.
Checkpoint and Llama revisions must be supplied from actual downloaded snapshots;
this document does not invent or recommend an unchecked weight revision.

- [Official model API](https://github.com/earthspecies/NatureLM-audio/blob/a4c65ad7d558f902852e20a299eafe927ca0921d/NatureLM/models/NatureLM.py):
  `NatureLM` inherits `PyTorchModelHubMixin`; `generate(samples, generate_cfg,
  prompts)` returns a list of decoded strings. The model combines BEATs,
  Q-Former, and a local Llama model. `init_audio_Qformer` uses BERT configuration,
  not pretrained BERT weights.
- [Official processor](https://github.com/earthspecies/NatureLM-audio/blob/a4c65ad7d558f902852e20a299eafe927ca0921d/NatureLM/processors.py):
  `NatureLMAudioProcessor` adds the audio placeholder/chat template, averages
  channels, resamples to 16 kHz, clamps amplitudes, and pads/truncates to 10 s.
- [Official pipeline](https://github.com/earthspecies/NatureLM-audio/blob/a4c65ad7d558f902852e20a299eafe927ca0921d/NatureLM/infer.py)
  demonstrates processor -> sample dictionary -> `generate`. Its window helper
  can omit a trailing partial window. This adapter explicitly includes that tail.
- [Official model card](https://huggingface.co/EarthSpeciesProject/NatureLM-audio)
  names Xeno-canto, iNaturalist, and **Watkins** among training data. Exact sample
  overlap with these registered test rows has not been audited. Do not describe
  Watkins as a verified unseen-data evaluation of this specialist.

## Local assets and invocation

Install the official source and its pinned environment separately. There is no
download or model load at module import. `NatureLMBackend(assets)` validates and
hashes local files before constructing the official model. Required dictionary:

```python
assets = {
    "naturelm_code": "/local/NatureLM-audio",  # clean git checkout
    "naturelm_source_revision": "a4c65ad7d558f902852e20a299eafe927ca0921d",
    "naturelm_checkpoint": "/local/naturelm-hf-snapshot",  # config.json + model.safetensors
    "naturelm_checkpoint_revision": "<actual 40-character downloaded HF revision>",
    "naturelm_llama": "/local/llama-3.1-8b-instruct-snapshot",
    "naturelm_llama_revision": "<actual 40-character downloaded HF revision>",
    "naturelm_bert_config": "/local/bert-base-uncased/config.json",
    "naturelm_config": "/local/pinned-inference.yml",
    "naturelm_device": "cuda",  # optional; explicit cpu is allowed for feasibility
    "naturelm_max_new_tokens": 32,
    "naturelm_merging_alpha": 1.0,
}
from akr_final.train50_naturelm import NatureLMBackend
backend = NatureLMBackend(assets)
result = backend.predict(query, registered_prompt, registered_labels)
```

The Llama directory needs config.json, tokenizer_config.json, tokenizer.json or
tokenizer.model, and model.safetensors or all shards in its safetensors index.
NatureLM supports the same indexed form. Supply the official `bert-base-uncased`
config locally. A subclass changes only its remote config lookup to
`BertConfig.from_json_file(local_path)`; Q-Former initialization and model classes
remain official. All Llama/NatureLM loads use explicit local directories; the
NatureLM HF loader receives `local_files_only=True`. The local inference YAML
must contain the official `generate` fields. The adapter fixes `do_sample=False`,
the declared output limit, and merge alpha; it records the effective remaining
generation configuration from the YAML. Alpha 1 is the full pretrained specialist,
not a task-selected merge. The current C15 runner loads the generalist first,
closes it, then loads NatureLM in the same process. It does not implement a
specialist subprocess. Use a dedicated C15 environment whose dependencies can
serve both backends; production acceptance of that joint environment is pending.
The inspected official `pyproject.toml` requires `transformers[sentencepiece]>=4.44.2`
and `peft>=0.11.1`, so its declared constraints do not exclude the Qwen pins
4.52.3/0.17.1. This does not establish runtime/API compatibility. Official Torch
and torchaudio constraints are >=2.6.0,<2.7, torchvision >=0.21.0,<0.22,
NumPy >=1.26.4, Pydantic >=2.7.4 and pydantic-settings >=2.7.1. A runtime conflict
must remain a failed/pending case until resolved, rather than silently changing
registered packages or substituting outputs. Separate-process orchestration would
be an additional implementation if required.

Provenance records SHA-256 and size for an explicit implementation-file list,
NatureLM and Llama weight files (or indexed shards), configs, and tokenizer files.
It does not recursively hash unrelated audio/cache/results directories. Source
HEAD and tracked changes are checked; supplied weight revision strings are
declared download provenance, while recorded hashes identify actual loaded local
bytes. Independently verify their official origin when provisioning assets.
Checkpoint, YAML, code, or tokenizer changes require a new case/run identity.

## Input, windowing, and normalization

`predict` accepts only the label-free `akr_closing.core.Query`, the exact registered
prompt, and the fixed label vocabulary. It does not receive labels for the query,
support demonstrations, a training manifest, or task fitting data. It uses the
Query audio path from the same immutable registration as the native generalist.
The case runner remains responsible for comparing IDs/hashes against registration
and requiring complete official test coverage; this backend must not choose rows.

Every audio sample is covered by nonoverlapping 10 s windows; a final/short window
is processed with the official padding behavior. Each raw answer is retained.
The repository's `animal_omni.metrics.normalize_label` normalizes each answer.
All windows must give the same valid label for an overall valid prediction;
disagreement or any invalid answer produces an invalid prediction. This explicit,
conservative aggregation is an adapter protocol, not an official NatureLM
leaderboard metric. It must be frozen before seeing test scores. Returned output
contains window intervals, raw text, audio/prompt/vocabulary hashes, and backend
provenance. Invalids remain invalid; no fallback is substituted.

## Scientific scope

C15 runs frozen **native specialist** alongside frozen native generalist on the
same registered full official tests for Dogs, Watkins, Bats, and HumBugDB. The
suite's 50 stratified official-train examples are shared registration context;
this native specialist uses **zero task adaptation examples**. Its external
pretraining supervision is different from Train-50 and must be stated separately.
This is not a same-pretraining-data or same-total-supervision comparison.

Dogs is individual identity classification. Arbitrary individual names/IDs do not
have a useful pretrained semantic mapping. A species/breed-capable native model
cannot infer a new ID-to-call mapping merely from the ID vocabulary; a low or
invalid score therefore does not establish weak dog acoustics. No demonstrations,
renamed semantic classes, or test-dependent prompt changes are silently added.
For all four tasks, label-wording compatibility and exact pretraining overlap need
audit. Watkins membership in broad training data is explicitly known; exact clip
overlap and overlap for Dogs/Bats/HumBugDB remain unestablished.

No AKR K/V extraction/injection adapter is implemented for NatureLM. A successful
C15 native comparison cannot be described as AKR transfer to NatureLM's
BEATs/Q-Former/Llama architecture. That claim requires a separate implemented and
verified adapter, support-gradient/zero-update checks, and matched native/+AKR
results. Tiny injected-model tests have `validation=contract_test_injected` in
provenance and must never enter scientific result tables.
