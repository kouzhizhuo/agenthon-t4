# Agenthon Track 4

This repository exports the exact best submitted V2 runtime. Official Development submission 968359 finished at **0.3462**; V4 submission 968634 finished at **0.1222** and was rejected as a replacement. Program logs and per-unit details were withheld, so aggregate differences cannot identify their cause. `results/official-feedback.json` records the public measurements.

V2 batches four entities per House request and allows one structural repair per batch, with a complete-roster contract. `runtime/house_analyze.py` and `runtime/analyze.py` are byte-for-byte copies of the frozen submitted V2. The runtime needs only Python's standard library. The published V2 image used Python 3.13 on Linux. The Dockerfile retains its pinned base and adjusts only COPY paths for this repository layout; this export has not built a new image.

Run against organizer-provided input:

```sh
python runtime/house_analyze.py analyze --task /input/task.json --corpus /input/corpus --out /output/answer.json
```

The official runner injects its House endpoint/model/token. Credentials and team proofs are not part of this repository. `requirements-validation.txt` lists optional local validation tools; the runtime has no third-party package dependency.

The `research/` documents retain compact rejected experiments and protocol-only next hypotheses. V5 reduced MAE but hurt ranking; V6 failed source cadence eligibility; V7 scheduler remains on hold pending its publication-bound clarification; V8 failed its assumed ownership metadata contract before code; V9 contextual source binding has only a protocol and inventory bounds. No V5–V9 replacement was submitted. Full preserved experiments, failure logs and data inventories remain in the original workspace and are referenced by hash in `research/DECISIONS.json`.

The source-contract diagnosis matters: organizer manifests are checksum ledgers, while issuer/series metadata is optional retrieval provenance. V2's evidence retrieval uses fallback documents and identity heuristics; public output claims fall back to task-row quotes when ownership labels are missing. Safely retained source/period/units and actual matched financial quality must be demonstrated before promoting a new method. Synthetic runtime checks are mechanical evidence.

`EXPORT_MANIFEST.json` maps every copied or derived export to its source hashes and exact file bytes. Raw corpora, target/label sets, credentials, OCI blobs and other tracks are excluded. This is a clean local export; repository creation and push are handled by the coordinator.
