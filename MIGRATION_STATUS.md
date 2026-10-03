# AKR consolidation scope and status

Date: 2026-10-03. This repository contains recovered executable research code and the supplied manuscript/results. Packaging completeness is not experimental success.

## Included

- Pinned public `Yunbo-max/animal-omni-kv@9e8c244b797c24cba1933998bbafbf44a36b0d86`: historical implementation, scripts, manifests, positive/mixed/negative results and protocol corrections.
- Available E1--E9, F1--F3 and no-old-lock update packages: `akr_closing`, `akr_final`, configuration, tests and scripts. Newer integration versions take precedence over older overlay files; conflicts/hashes are retained in `migration/COMPLETION_SOURCE_MAP_20261003.json`.
- The user-approved nine-page RQ-organized manuscript, complete bibliography, vector result figures, portable main-figure derivatives, completed scalar exports and figure prompt history.
- S1 same-support simplest-alternative implementation and a bounded release entrypoint. S1 has not been run on a real model in this release.
- Retrospective audit using the actual 2026-10-02 research-autopilot export; exact read-file hashes are in `docs/research/SKILL_USE_20261003.json`.

## Limits retained

The connection returned 404 for `yuhanlydia/animal` on 2026-10-03. Restoring supplied packages is not verification of that inaccessible repository's latest HEAD, branches or unprovided files. No source repository is deleted or overwritten.

Raw audio, foundation-model weights, LoRA checkpoints and large representation/gradient caches were not provided for this consolidation and are not claimed transferred. Fetch data under their original licenses or reuse local files. Missing caches can be recomputed; missing historical identities must not be invented. New standalone runs do not require old RUN.json or LOCK.json.

Full-resolution author PNGs are retained in the supplied Overleaf source archive; the repository manuscript uses the existing verified portable JPEG renditions. All rendition dimensions/hashes are in `migration/figure_renderings.json`. Internal author discussion snapshots and the private custom skill implementation are not published.

## Verification

Fresh local baseline: 137 CPU tests. After completion fixes: 143 CPU tests; 116 offline release checks. The manuscript compiles to 24 total pages, with main text ending on page 9. One underfull vertical-box layout warning remains; there are no undefined references, missing figures or overfull boxes. This does not certify GPU runtime, new model scores, legal/license clearance for redistribution of source datasets, or paper acceptance.

Remote integration uses the existing PR #1 and ordinary versioned source files. The final commit/PR and Actions status, rather than this document alone, establish whether the integrated branch has reached main.
