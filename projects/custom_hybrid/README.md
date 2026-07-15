# Custom Hybrid workflow

This project keeps MinerU's Hybrid layout/OCR pipeline while putting every vLLM
generation request behind a configurable OpenAI-compatible proxy. It separates:

- vLLM engine arguments, applied when the server starts;
- generation arguments, applied to `/chat/completions` requests;
- MinerU Hybrid behavior (`effort`, parse method, OCR language, formulas, tables);
- reproducible Markdown quality evaluation.

The parameter proxy is a native ASGI application using `uvicorn` and `httpx`; it
does not require FastAPI and forwards upstream response bytes as a stream.

## Prerequisites

Install the MinerU client/OCR environment from this checkout. Use the same Python
environment for the workflow command:

```powershell
python -m pip install -e ".[core]"
```

Run vLLM on a supported Linux GPU host (or use an existing OpenAI-compatible
MinerU server) and set `vllm.upstream_url` to its reachable base URL. To launch it
from a Linux MinerU environment, configure `vllm.server_args` and run the
`serve-vllm` command below. vLLM is not expected to run natively on Windows.

For an authenticated server, set the environment variable named by
`vllm.api_key_env`; the default is `VLLM_API_KEY`. Secrets are never written to
the workflow JSON or audit log.

The visual conflict verifier can reuse the extraction endpoint or use a separate
instruction-following vision endpoint through `fusion.verifier.base_url`. Build
reference Markdown with a strong model and manually review it before benchmarking;
an unverified model output is not reliable ground truth.

## 1. Configure

Copy `workflow.example.json` and edit the copy. Important sections:

- `vllm.server_args`: arguments accepted by `mineru-vllm-server`/vLLM;
- `vllm.generation.defaults`: values added only when MinerU did not send one;
- `vllm.generation.overrides`: values that always replace MinerU's request;
- `vllm.generation.remove`: MinerU/model defaults removed before forwarding;
- `vllm.generation.rules`: prompt/path/model regex-specific overrides;
- `mineru`: the OCR and Hybrid strategy used for extraction.

`generation_config` is set to `vllm` in the example so the model repository's
generation config does not silently replace the workflow's sampling baseline.
The example also removes MinerU's stale `top_k`, presence/frequency penalties,
and `vllm_xargs`, while capping generated output at `max_tokens=4096`. For a
remote server with `max_model_len=8192`, keep prompt tokens plus `max_tokens`
within 8192. Configure the OpenAI-compatible base URL (for example,
`http://10.100.0.30:8205`), not its interactive `/docs` page.

Check the local runtime and configured vLLM endpoint before extraction:

```powershell
python projects/custom_hybrid/workflow.py --config workflow.local.json doctor
```

The command reports missing proxy dependencies and probes `<upstream>/v1/models`.
The parameter proxy binds to loopback by default. A non-loopback host is rejected
unless `vllm.proxy.allow_public_bind=true`, because the proxy can carry model API
credentials and should not be exposed accidentally.

## 2. Start vLLM

```powershell
python projects/custom_hybrid/workflow.py --config workflow.local.json serve-vllm
```

The configured vLLM server listens on `vllm.upstream_url`. Engine arguments are
passed as an argument list, without shell evaluation.

## 3. Extract

```powershell
python projects/custom_hybrid/workflow.py --config workflow.local.json extract `
  --input demo/pdfs/small_ocr.pdf `
  --output output/custom_hybrid
```

With `fusion.enabled=true`, `extract` starts the local parameter proxy and runs
two independent parses:

- `output/hybrid`: Hybrid/VLM result with configured generation parameters;
- `output/ocr`: pipeline backend forced to OCR mode;
- `output/fused`: Hybrid structure with conservative OCR/VLM corrections and
  retained Pipeline table Cell/content geometry;
- `output/fusion_summary.json`: per-document status and replacement counts.

Each fused parse directory also contains `<document>_fusion.json`, which records
the bbox, candidates, confidence, similarity, and decision for every conflict.
The vLLM JSONL audit log contains applied parameter names, latency, status, and a
prompt hash. It does not store prompts, images, or authorization headers.
Set `vllm.audit_prompt_preview_chars` to a small positive value while developing
stage-specific regex rules. The preview collector excludes image/audio fields and
all `data:` URIs; leave it at `0` for normal runs.

Fusion only auto-replaces empty, corrupt, repeated, or abnormally expanded VLM
text when OCR confidence passes the threshold. Other disagreements are shown as
page crops to the configured visual verifier. The built-in verifier can only
select the exact `hybrid` or `ocr` candidate; it cannot transcribe a third value.
High-confidence OCR lines that have no Hybrid target can be inserted as recovered
text blocks, but lines inside tables/images/charts/formulas are excluded. Control
this with `recover_missing_ocr_blocks`, `missing_ocr_min_confidence`, and
`max_missing_ocr_blocks_per_document`. Equations and table bodies are not
rewritten by the text correction pass. Empty formulas still use the matched
Pipeline formula; conflicting non-empty formulas use conservative visual
candidate selection.

### Hierarchical table fusion

Tables use a global-structure/local-content/global-validation pipeline:

1. Spatially match each Pipeline table to one Hybrid table.
2. Parse both HTML candidates into logical grids, including `rowspan` and
   `colspan`. Hybrid remains the structural backbone.
3. Enable cell fusion only when row/column dimensions and every logical cell
   range match exactly. A mismatch falls back to whole-table selection.
4. Use Pipeline `table_cells` metadata for exact page-level cell bboxes. Each
   Cell can additionally contain `content_bbox`, OCR `content_spans` with bbox and
   polygon, and a weighted OCR `confidence`. Metadata text must agree with the
   corresponding Pipeline HTML cell before its geometry or confidence is trusted.
5. Keep consensus cells, fill empty Hybrid cells, and replace suspicious Hybrid
   cells only when Pipeline supplies sufficient OCR confidence (or the explicit
   `table_cell_allow_unscored_ocr` opt-in is enabled).
6. Send remaining conflicts to the visual verifier with four possible crops:
   whole table, target row, target column, and target cell. The verifier may only
   choose `hybrid` or `pipeline`; it cannot invent or merge a third value.
7. Replace only the selected cell's inner HTML, then parse the rebuilt table and
   require its structure signature to remain identical to Hybrid. Unsafe cell
   HTML, invalid reconstruction, verifier errors, missing bboxes, and request
   limits all keep Hybrid.

The fused table copies Pipeline Cell geometry into the final Hybrid table and
synchronizes every Cell's `text` with the final fused HTML. Bounding-box PDFs draw
Cell boundaries in orange and OCR content-tight boxes in cyan. If OCR confidence
is absent, the safe default `table_cell_allow_unscored_ocr=false` sends conflicts
to visual verification instead of auto-replacing them.

Important table controls:

- `table_cell_fusion_enabled`: turn hierarchical cell fusion on or off;
- `table_cell_consensus_similarity`: skip already-agreeing cells;
- `table_cell_min_ocr_confidence`: automatic suspicious-cell replacement gate;
- `table_cell_metadata_text_similarity`: reject misaligned cell metadata;
- `max_table_cells_per_table`: avoid unbounded parsing/verification work;
- `max_table_cell_verifications_per_document`: cap cell-level VLM calls;
- `verifier.table_cell_render_scale`: render table crops separately at higher
  PDF resolution (`4.0` is approximately 288 DPI);
- `verifier.table_cell_include_*_image`: control whole-table, row, and column
  context images.

The per-document `_fusion.json` records every table structure decision and every
cell candidate, bbox, similarity, confidence, selected source, verifier failure,
and reconstruction rejection. Summary counters distinguish cell consensus,
empty/suspicious/visual replacements, missing bboxes, structure mismatches, and
whole-table fallbacks.

`fusion.verifier.base_url=null` reuses the extraction vLLM. For more reliable
arbitration, point it at an instruction-following vision model served through an
OpenAI-compatible endpoint and set `fusion.verifier.model`. The MinerU-specialized
1.2B extraction model may not consistently follow the custom JSON arbitration
prompt; failures are recorded and conservatively keep the Hybrid candidate.

After changing fusion thresholds, reuse existing Hybrid and OCR results:

```powershell
python projects/custom_hybrid/workflow.py --config workflow.local.json fuse `
  --input demo/pdfs/small_ocr.pdf `
  --hybrid-root output/custom_hybrid/hybrid `
  --ocr-root output/custom_hybrid/ocr `
  --output output/custom_hybrid/fused-tuned
```

For a long-running MinerU API, run the proxy separately and point
`hybrid-http-client` at it:

```powershell
python projects/custom_hybrid/workflow.py --config workflow.local.json proxy
```

## 4. Serve the fused workflow over HTTP

The standard `mineru-api` does not run the dual Hybrid + Pipeline OCR fusion
workflow. Use the dedicated service to upload documents and download a ZIP whose
root contains `fused/`, `fusion_summary.json`, and the per-task vLLM audit log.

Set a bearer token before binding the service to a public/container interface:

```bash
export CUSTOM_HYBRID_API_KEY='replace-with-a-long-random-token'

python projects/custom_hybrid/api.py \
  --config workflow.local.json \
  --host 0.0.0.0 \
  --port 8010 \
  --output-root /work/mineru-output/custom-hybrid-api
```

Public binding without a token is rejected unless
`--allow-unauthenticated-public` is explicitly supplied. Keep the service behind
a private network, VPN, or authenticated reverse proxy. The server uses the
generation and upstream settings from `workflow.local.json`; request clients
cannot override them. Tasks run serially because each extraction owns a local
parameter-proxy port and local OCR resources.

Endpoints:

- `GET /health`: service status;
- `POST /tasks`: upload PDF/images and receive a task id;
- `GET /tasks/{task_id}`: poll status;
- `GET /tasks/{task_id}/result`: download the fused ZIP;
- `GET /tasks/{task_id}/report`: read `fusion_summary.json`;
- `DELETE /tasks/{task_id}`: remove a completed/failed task and its files;
- `POST /file_parse`: wait synchronously and return the fused ZIP.

From a Mac, only Python and `httpx` are required. Run the client from this repo:

```bash
python -m pip install httpx
export CUSTOM_HYBRID_API_KEY='replace-with-a-long-random-token'

python projects/custom_hybrid/api_client.py \
  --url http://10.100.0.30:8010 \
  --input ~/Documents/invoice.pdf \
  --output ~/Documents/invoice-fused.zip
```

The client accepts either one supported file or a directory and streams the ZIP
to disk. The Jupyter/container port must be published or reverse-proxied before a
Mac can reach it. Task state is in memory and does not survive a service restart;
task files remain under `--output-root` until deleted.

If `jupyter-server-proxy` is installed and direct port publishing is unavailable,
use `http://<jupyter-host>:<jupyter-port>/proxy/8010` as `--url` and export the
Jupyter access token through `JUPYTER_TOKEN`. The client sends that token as a
query parameter while retaining the Custom Hybrid bearer token in the
`Authorization` header.

## 5. Run the lightweight local UI

The local UI is a small FastAPI/HTML application. It runs on the Mac, keeps API
credentials out of browser JavaScript, and proxies uploads, task polling, reports,
and ZIP downloads to the remote Custom Hybrid service. It does not install or run
MinerU models locally.

```bash
python -m pip install fastapi uvicorn httpx python-multipart
export CUSTOM_HYBRID_API_KEY='the-same-token-used-by-the-server'

python projects/custom_hybrid/ui.py \
  --remote-url http://10.100.0.30:6108 \
  --host 127.0.0.1 \
  --port 7860
```

Open `http://127.0.0.1:7860`, drag in PDF/images, submit the task, inspect the
fusion report, and download the fused ZIP. The UI refuses public binding unless
`--allow-public-bind` is supplied explicitly.

For a Jupyter Server Proxy URL, also export `JUPYTER_TOKEN` and use a remote URL
such as `http://10.100.0.30:8989/proxy/6108`.

## 6. Evaluate

Create reference Markdown manually or with a strong model, then compare either
two files or directory trees with matching `.md` filenames:

```powershell
python projects/custom_hybrid/workflow.py --config workflow.local.json evaluate `
  --reference benchmarks/reference `
  --candidate output/custom_hybrid `
  --report output/custom_hybrid/report.json
```

The report includes character error rate, token F1, line-order similarity,
Markdown structure F1, and a weighted quality score. It also measures table count
F1, logical-grid structure F1, exact cell match, cell text similarity, and a
separate table quality score for HTML and Markdown pipe tables. Keep a fixed benchmark set
covering native PDFs, scans, mixed Chinese/English, formulas, tables, multi-column
pages, headers/footers, and poor-quality images. Parameter changes should be kept
only when they improve the aggregate score without creating document-level
regressions.

For repeatable experiments, copy `benchmark.example.json`, create the referenced
GPT/manual Markdown files, and rank several output trees in one report:

```powershell
python projects/custom_hybrid/workflow.py --config workflow.local.json benchmark `
  --manifest benchmark.local.json `
  --candidate hybrid=output/custom_hybrid/hybrid `
  --candidate fused=output/custom_hybrid/fused `
  --report output/custom_hybrid/benchmark.json
```

The benchmark report uses document weights and includes per-tag aggregates, so a
gain on ordinary native PDFs cannot hide regressions on scans, formulas, tables,
or multi-column reading order. The first `--candidate` is the baseline; later
runs include aggregate, category, and per-document deltas plus a regression list.
Use `evaluation.regression_tolerance` to ignore insignificant score noise.
Missing candidate documents reduce weighted coverage and therefore the leaderboard's
coverage-adjusted quality score; missing reference documents fail the benchmark.

Run the configured generation grid with one shared OCR parse and optional scoring:

```powershell
python projects/custom_hybrid/workflow.py --config workflow.local.json sweep `
  --input benchmark/documents `
  --output output/custom_hybrid/sweep-001 `
  --manifest benchmark.local.json
```

Each `run-NNN` directory records the exact workflow JSON and vLLM request audit.
The final `sweep_summary.json` contains failures, fusion statistics, per-category
scores, and a leaderboard. Keep grids small; decoding parameters cannot compensate
for a weak visual model, insufficient render resolution, or wrong reading order.

## Accuracy notes

Low-temperature deterministic decoding is a good extraction baseline, not a
guarantee of correctness. `max_tokens` prevents truncation but cannot recover
information outside the model's visual resolution/context. OCR language, scan
resolution, layout routing, model capability, and post-processing usually have
more impact than small `top_p` changes. A GPT-like target therefore needs a
representative reference set and per-category error analysis.
