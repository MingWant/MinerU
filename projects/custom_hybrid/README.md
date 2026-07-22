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

No authorization header is sent by default. For an authenticated server, set
`vllm.api_key_env` (or the corresponding recognizer/recovery/verifier setting)
to the name of an environment variable containing the token. Secrets are never
written to the workflow JSON or audit log.

The visual conflict verifier can reuse the extraction endpoint or use a separate
instruction-following vision endpoint through `fusion.verifier.base_url`. Build
reference Markdown with a strong model and manually review it before benchmarking;
an unverified model output is not reliable ground truth.

## 1. Configure

Copy `workflow.example.json` and edit the copy. Important sections:

- `vllm.server_args`: arguments accepted by `mineru-vllm-server`/vLLM;
- `vllm.generation.defaults`: values added only when MinerU did not send one;
- `vllm.generation.overrides`: values that always replace MinerU's request;
- `vllm.generation.context_reserve_tokens`: context kept available for prompts;
- `vllm.generation.max_context_tokens`: optional manual context limit when the
  upstream `/v1/models` response does not expose `max_model_len`;
- `vllm.generation.remove`: MinerU/model defaults removed before forwarding;
- `vllm.generation.rules`: prompt/path/model regex-specific overrides;
- `mineru`: the OCR and Hybrid strategy used for extraction.

`generation_config` is set to `vllm` in the example so the model repository's
generation config does not silently replace the workflow's sampling baseline.
The example is cost-optimized by default: Hybrid uses `effort=medium`, formula
and image analysis are disabled, and the optional visual verifier, recognizer,
and reconciliation stages are off. Pipeline OCR, deterministic fusion, Table
processing, Cell metadata, and content-tight bbox rendering remain enabled.
An existing `workflow.local.json` is not rewritten during an update. API/UI tasks
still receive the Balanced runtime overrides by default; direct `extract` users
should merge these example values into their local config explicitly.
The example also removes MinerU's stale `top_k`, presence/frequency penalties,
and `vllm_xargs`, while overriding generated output to `max_tokens=2048` instead
of using a default that MinerU's own `max_tokens` can bypass. At proxy startup,
the workflow reads `/v1/models` and clamps requested output tokens to
`max_model_len - context_reserve_tokens`; this guard runs after task-level UI
overrides. For a remote server with `max_model_len=8192`, the example reserves
4096 tokens for the prompt and therefore never forwards more than 4096 output
tokens. Configure the OpenAI-compatible base URL (for example,
`http://10.100.0.30:8205`), not its interactive `/docs` page.

Check the local runtime and configured vLLM endpoint before extraction:

```powershell
python projects/custom_hybrid/workflow.py --config workflow.local.json doctor
```

The command reports missing proxy dependencies and probes `<upstream>/v1/models`.
When `fusion.recognizer.enabled=true`, it also checks the independent Vision
endpoint, configured model ID, context length, and whether the selected strict
output mode appears in its OpenAPI schema. `recognizer.ready=true` proves only
transport and protocol compatibility; `capability_trial_required=true` remains
because bbox-ground-truth A/B must still prove transcription quality.
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

With `fusion.enabled=true`, `extract` starts the local parameter proxy. The
default `fusion.mode=hybrid_fusion` runs two independent parses:

- `output/hybrid`: Hybrid/VLM result with configured generation parameters;
- `output/ocr`: pipeline backend forced to OCR mode;
- `output/fused`: Hybrid structure with conservative OCR/VLM corrections and
  retained Pipeline table Cell/content geometry;
- `output/fusion_summary.json`: per-document status and replacement counts.

The alternative `fusion.mode=bbox_vlm` is a Pipeline-owned geometry path. It
runs only `output/ocr`, checks and repairs the OCR content bboxes first, then
sends the complete repaired bbox set as local crops to the constrained Vision
recognizer, and writes selected text into `output/fused`. The OCR text is retained
when the VLM response is missing, malformed, unsafe, or fails a structured-value
guard. Repair is part of this same OCR post-processing pipeline: Table/Cell
geometry and the Table grid remain immutable while missing or incomplete content
boxes are added or adjusted before recognition.

When `fusion.semantic_markdown.enabled=true`, the fused middle JSON is also
rendered into a deterministic semantic Markdown view (`<document>.md`). The
renderer uses OCR span geometry, Cell row/column metadata, and conservative
field-type checks rather than document-specific names or coordinates. It pairs
labels with values in the same Cell or an adjacent value Cell, joins split
checkbox/list markers, rebuilds ledger tables, removes repeated page metadata,
and normalizes common dates and amounts. Version 5 treats Cell text as the
authority when an OCR span crosses neighboring Cells, anchors repeated
policy-owner/insured and physician signature grids by row and column, and emits
diagnostic, treatment, laboratory, and follow-up grids as Markdown tables instead
of flattening their columns. It also preserves fragment-heavy pages with an
explicit warning, recursively recovers text from demoted/unstructured Table
blocks, requires spatial evidence for de-duplication, and emits a
`<document>_semantic_report.json` coverage sidecar. The report distinguishes
text-bearing pages from blank/image-only pages and records represented,
schema-normalized, de-duplicated, filtered, and unmatched source evidence. The
label vocabulary covers common claim, member, provider, contact, identifier,
address, date, email, postal-code, and amount fields, so new form layouts do not
require PDF-specific rules. Set
`preserve_native=true` to keep `<document>_native.md` for direct A/B comparison
with MinerU's original Markdown. The output marker `semantic-markdown-v5`
identifies this renderer version.

Replay one or more fused middle JSON files without calling MinerU or vLLM:

```powershell
python projects/custom_hybrid/semantic_markdown_benchmark.py `
  output/fused `
  --output output/semantic_markdown_benchmark.json
```

The benchmark summary reports total and emitted text-bearing pages, generation
failures, fragment-heavy pages, unstructured Table fallbacks, and unmatched
source records. Per-document reports also identify the page numbers containing
unstructured Table fallbacks. Each benchmark document entry includes only the
unmatched trace details; the per-document semantic report retains the complete
source trace.

Each fused parse directory also contains `<document>_fusion.json`, which records
the bbox, candidates, confidence, similarity, and decision for every conflict.
The vLLM JSONL audit log contains applied parameter names, latency, status, and a
prompt hash. It does not store prompts, images, or authorization headers.
Set `vllm.audit_prompt_preview_chars` to a small positive value while developing
stage-specific regex rules. The preview collector excludes image/audio fields and
all `data:` URIs; leave it at `0` for normal runs.

Fusion only auto-replaces empty, corrupt, repeated, or abnormally expanded VLM
text when OCR confidence passes the threshold. Other disagreements keep Hybrid
by default. When explicitly enabled, the visual verifier receives page crops and
can only select the exact `hybrid` or `ocr` candidate; it cannot transcribe a
third value.
High-confidence OCR lines that have no Hybrid target can be inserted as recovered
text blocks. Images, charts, formulas, and geometrically reliable Tables remain
protected visual containers. An unreliable Table is routed to the coverage-first
path instead: OCR `content_spans` become bbox-backed `table_ocr` evidence, and
explicit labels such as `Policy No.` or `日期` can be paired with nearby values as
`form_field` records. Unscored Table OCR is allowed only through the explicit
`unreliable_table_allow_unscored_ocr` switch. General missing OCR and unreliable
Table OCR have separate document budgets, so a large claim form cannot consume
the ordinary paragraph-recovery allowance.

Before inserting Table OCR as text, the workflow performs count-aware structured
coverage matching against the final Hybrid Table HTML. OCR occurrences already
represented by Cell text count as covered and are not duplicated; only remaining
bbox-backed occurrences are inserted. Repeated values are consumed one occurrence
at a time instead of being globally suppressed by text equality.

Every fusion report includes per-page `coverage` records, global
`ocr_spatial_coverage`, Table quality routes, and structured `key_value_pairs`
with `key_bbox` and `value_bbox`. The fused middle JSON stores the same page-level
`form_fields` metadata. Bounding-box PDFs use cyan for OCR content boxes, green
for paired keys, and blue for paired values. Table Cell geometry remains in
middle JSON for fusion and downstream consumers, but is not drawn because the
content-tight cyan boxes are more useful for visual review.

Important coverage controls are `recover_missing_ocr_blocks`,
`missing_ocr_min_confidence`, `max_missing_ocr_blocks_per_document`,
`unreliable_table_recovery_enabled`,
`unreliable_table_allow_unscored_ocr`, and
`max_unreliable_table_ocr_blocks_per_document`.

### Experimental bbox-conditioned VLM recognition

Set `fusion.mode=bbox_vlm` to use the complete ordered bbox pipeline. The
recognizer is forced on, ordinary page-text recognition is forced off, and Table
content-box recognition is forced on. Immediately after OCR, the workflow
inspects Table Cells for visible ink that is missing from, or extends beyond,
existing content boxes. Blank Cells do not trigger a model request. Suspicious
Tables are rendered once with gray Cell borders, cyan existing content boxes,
and red suspicious Cells. A constrained Vision reviewer may return only `add` or
`adjust` proposals for known Cell/box IDs using rough normalized Table
coordinates. Repaired boxes are then included in the same recognition request;
there is no separate recovery extraction mode.

Reviewer coordinates never become final geometry directly. The workflow clips
them to the immutable Cell boundary, refines them against dark document pixels,
then applies confidence, area-ratio, IoU deduplication, adjustment-overlap,
per-Table, and per-document gates. Two deterministic local cases may instead be
clipped to the immutable Table boundary: one content line detected across
overlapping adjacent Cells, and signature/date ink extending below a truncated
terminal-row Cell. Table bounds, Cell bounds, row/column indices, and HTML grid
signatures are snapshotted; any invariant failure rolls back every accepted
proposal on that page. Accepted empty recovered boxes can enter local VLM
transcription only when their geometry confidence passes
`recognizer.recovered_empty_min_confidence`.

Recovery cost is bounded by `recovery.max_tables_per_document`,
`max_requests_per_document`, `max_proposals_per_document`, and
`max_proposals_per_table`. API `balanced` defaults to three reviewed Tables,
two hundred document proposals, and fifty proposals per Table; `quality`
defaults to ten Tables, five hundred document proposals, and one hundred per
Table. One Cell may yield several content-tight line boxes.
The English UI exposes optional overrides under **Advanced BBox Recovery
Settings**. Fusion reports record reviewed Tables, requests, accepted/added/
adjusted/rejected proposals, reviewer batches, decisions, and the Table/Cell
geometry invariant.

Recovery first handles Cells that visibly contain ink but have no content bbox
with deterministic local pixel refinement. These proposals require no VLM
request and are still subject to the normal confidence, area, duplicate, Cell,
and Table-grid safety gates. When the configured model name contains `mineru`,
geometry-review chat requests are skipped because MinerU extraction checkpoints
use their own constrained decoding protocol and do not reliably emit arbitrary
JSON coordinate schemas. An invalid structured response also opens a protocol
circuit breaker so later Tables retain the local/Pipeline geometry without
repeating expensive failed requests.

`fusion.recognizer.enabled=false` is the default. When explicitly enabled, the
recognizer runs before structured fusion and treats Pipeline OCR geometry as
immutable: the Vision model receives existing bbox IDs and crops, then returns
only `{id, text}` items. Unknown IDs, duplicate IDs, returned coordinates, and
free-form output are ignored. Selected text is written back into OCR spans and
Cell metadata without changing any bbox; keyed Pipeline Cells can safely rebuild
their HTML while requiring the Table structure signature to stay identical.

The recognizer supports target, row, column, and whole-Table crops. Target crops
always receive image-budget priority. Row context is enabled by default; column
and Table images are opt-in because they increase multimodal context cost. Batch
size, image/request limits, rendering scale, padding, timeout, model, endpoint,
sampling values, output tokens, and context reserve are server-controlled under
`fusion.recognizer`. The default image limit is eight per request. Candidate
batches are packed using the actual target images plus deduplicated row, column,
and Table contexts, so enabled context images are retained instead of being
silently displaced by target crops. If the server still rejects a request with
an `At most N image(s)` error, the recognizer learns that limit for the failed
batch, repartitions it, and retries up to `max_image_limit_retries`. Other HTTP
400 responses are not retried blindly.

`recognizer.protocol=auto` detects MinerU model IDs and switches to the official
single-crop `Text Recognition:` protocol. It sends only empty or visually tall
targets by default (`native_min_bbox_height=16`), preserving small printed OCR
labels and focusing requests on handwriting and filled values. Native responses
are raw text rather than JSON, use a 512-token stage cap, and never request row,
column, or whole-Table images. This avoids the failure mode where a MinerU model
generates thousands of tokens while attempting to satisfy an incompatible JSON
schema. Set `protocol=structured` for a general instruction-following VLM or
`protocol=mineru_native` to force the native path.

Native requests run in bounded waves (`native_max_concurrency=2` by default).
The Balanced API profile prioritizes empty and taller value crops, caps each
page at 12 network requests and each document at 50, uses render scale 3 and
JPEG quality 85, and limits native output to 256 tokens. Quality uses the wider
16-point height threshold, up to 24 requests per page, render scale 4, and a
512-token cap. Repeated native failures open a circuit breaker before later
waves are submitted.

`native_cache_enabled=true` stores successful native transcriptions in a
content-addressed cache keyed by model, prompt, token cap, and exact rendered
crop. Cache hits do not consume request budgets. Files are created with private
directory/file permissions and expire after `native_cache_ttl_seconds` (30 days
in the example). The default path is
`~/.cache/mineru/custom-hybrid/native-recognition`; it contains extracted text,
so deployments with stricter data-retention requirements should disable it or
point it at an encrypted/ephemeral volume.

For MinerU local-only Recovery, every eligible Cell is checked after Pipeline
OCR, including Cells that already have some content boxes. Existing OCR boxes
are padded and masked, then only meaningful uncovered ink is proposed as one or
more content-tight line boxes. Grayscale masks are evaluated with NumPy instead
of per-pixel Python loops. Long horizontal, vertical, and cross-Cell diagonal
Table rules are removed before the content-tight bbox is calculated. Highly
overlapping proposals owned by different Cells in the same Table are rejected
as duplicates; this protects against malformed OCR grids that map the same
printed row into several Cells.
When adjacent Cells detect vertically overlapping fragments of the same ink
line, `merge_split_content_boxes_enabled=true` combines them before fusion if
horizontal coverage, vertical overlap, and union-height guards pass. The merged
bbox is sent to vLLM once and is auditable through its source Cell IDs. The
guards are controlled by `split_content_min_horizontal_overlap`,
`split_content_min_vertical_overlap`, and
`split_content_max_union_height_ratio`.

For the final Table row only, labels such as signature, ID/passport number, and
date enable a bounded scan down to the existing Table bottom. This recovers
handwriting that lies below a truncated Cell without modifying the Cell or Table
grid. `terminal_field_min_bottom_extension` and
`terminal_field_max_bottom_extension` constrain the allowed geometry gap; the
extension is never enabled from a remote reviewer response. A locally recovered
date crop may contain a small printed-label fragment; when it contains exactly
one valid numeric date, that date is extracted before the normal empty-OCR
capacity and safety guards are applied.
Full-width Form rows ending in `From`, or containing a bounded From/To or 由/至
date range, receive a separate `date_range_field_bottom_extension` scan. This
captures low `Day / Month / Year` legends whose OCR box was clipped at the Cell
bottom. Only locally observed ink may cross that Cell boundary, it remains
inside the detected Form/Table region, and
`form_full_cell_recovery_max_width_ratio` limits overly broad proposals.
Reports and the English UI expose analyzed/skipped Cell counts, pixel-analysis
time, cache hits/misses, request-budget skips, and native network requests.

In `bbox_vlm` mode, the bbox repair stage reuses the recognizer's rendered-page
cache by default (`share_recognizer_page_cache=true`). It performs its own
Cell-local ink analysis and never changes Table/Cell geometry, but it no longer
renders the same PDF page a second time before native transcription.

`table_orphan_recovery_enabled=true` also scans only the part of each Table that
is outside the union of all OCR Cell bboxes. After removing long Table rules and
borders, content-like line bands are inserted as `add_orphan` content boxes and
sent through the same native transcription path. This covers truncated grids
where visible text remains inside the Table but below or beside every detected
Cell. Orphan boxes may be attached to the nearest Cell for text/HTML ownership,
but the original Table and Cell bboxes remain immutable. If the per-Table orphan
budget is full, more substantial text-line candidates are preferred over thin
ink fragments; `table_orphan_preferred_line_height` controls that ranking bias.

`table_fringe_recovery_enabled=true` extends that masked scan a bounded distance
below a detected Table. It recovers labels and dates that visually belong to the
form but fell just outside the Pipeline Table boundary, without expanding or
moving the Table/Cell geometry.

`checkbox_recovery_enabled=true` runs a separate local contour detector for
small square form controls. It accepts both empty and tick-connected outlines,
requires four-sided border evidence, nearby label ink, and a short whitespace
separator before that label. The separator rejects square-looking capital
letters such as D/Q without rejecting a tick that protrudes beyond its box. The
ambiguous path additionally requires a standard four-corner outline, rejecting
handwritten box-like glyphs. The detector ignores tight OCR boxes that already
cover the control and attaches a new content-tight cyan box to the containing/
nearest Cell. Clear states are stored locally as checked or unchecked and do
not require a model request; only ambiguous interiors retain empty OCR text and
enter the normal bounded native-recognition queue. When the checkbox and its
right-hand label were split only by OCR spacing, `checkbox_merge_label_enabled`
updates the label content bbox to cover both; a checkbox already covered by a
label bbox does not create a redundant square-only box.
For malformed checks whose stroke extends far outside a very small square, the
normal `checkbox_min_size` remains unchanged. A lower
`checkbox_protruding_tick_min_size` is used only when a bounded multi-vertex
stroke covers most of the square and passes configured width, height, and
coverage limits. When merged with its label, the final bbox includes that full
stroke even if it crosses the original Cell edge, while remaining clipped to the
immutable Table boundary.

`list_marker_merge_enabled=true` applies the same post-OCR grouping principle
to numbered list markers. A strict standalone marker such as `1.`, `2)`, or
`(3)` is merged only with the nearest right-hand text bbox in the same Cell when
their line overlap and gap pass the configured guards. The marker's original
bbox remains in middle JSON for audit but is hidden from visualization and
recognition; the combined bbox and fallback text (for example,
`1. ContentABCDEFG`) are sent to vLLM. Amount-only, code-like, distant, and
vertically misaligned neighbors are not merged.

`ink_marker_merge_enabled=true` covers the corresponding case where a
handwritten circled/list marker has no OCR span at all. For visually tall text
boxes, a bounded masked scan looks immediately to the left for a compact,
same-line ink prefix. A guarded match expands the existing content bbox over the
marker and sends the combined crop to vLLM once. Printed labels, thin strokes,
distant ink, and markers already covered by another bbox remain excluded.

`structured_output_mode=json_schema` is the default example and sends a dynamic
strict schema whose ID enum and item count match the current batch. vLLM-native
`structured_outputs`, strict ordered `regex`, legacy `json_object`, and `none`
are available explicitly for compatible servers. Native schema mode also asks
vLLM to disable unconstrained whitespace. Regex mode encodes every batch ID and
JSON-string escape rules directly, leaving no whitespace escape path.
Constrained modes hide real bbox IDs from the prompt and expose only ordinal
image slots, preventing protocol-token copying; returned bbox-ID-shaped text is
always rejected as an echo and falls back to OCR.
Unsupported strict modes fail closed and retain OCR.
After repeated invalid structured batches, `max_consecutive_invalid_schema`
opens a document-level circuit breaker and skips the remaining recognition
requests. Stage-specific token caps remain effective even when a task-level
`max_tokens` override is larger.

Candidate selection is conservative:

- normalized consensus keeps OCR;
- invalid OCR dates/amounts may use a format-valid VLM transcription;
- when two valid dates, amounts, or identifiers disagree, OCR is retained and
  the high-risk conflict is audited;
- unrelated text and abnormal length changes are rejected;
- low-similarity jumps between strongly Latin and strongly CJK text are rejected,
  including in `vlm_primary` mode; mixed bilingual fields and empty-BBox recovery
  are unaffected (`script_guard_enabled=true` by default);
- high-confidence OCR stays unless it is demonstrably lower quality;
- unscored OCR may prefer VLM only when all guards pass.
- batches with fewer than the configured proportion of plausible responses
  trigger a circuit breaker and retain OCR for the entire batch; the original
  per-candidate reason remains in `candidate_reason` for diagnosis.

Existing OCR/content bboxes with empty text are retained as recognition
candidates when `empty_ocr_enabled=true`. An empty candidate can be filled only
when a batch of at least `batch_guard_min_candidates` passes the configured
quality ratio using non-empty OCR anchors. Isolated empty boxes and low-quality
batches remain empty and are audited as `empty_ocr_context_guard`.
Very thin single-line recovery boxes also apply a stricter physical text-density
limit (`empty_thin_line_max_chars_per_em`, default `2.5`), preventing a tiny ink
fragment from accepting an implausible sentence. The stricter limit applies only
to local boxes up to 240 page units wide, leaving full-width printed lines and
taller handwriting regions on the normal geometry budget.
Repeated short phrases in an empty recovery candidate are also rejected as
`empty_ocr_repetition_guard`; this catches compact looped hallucinations that do
not exceed the geometry limit.
Accepted recovery boxes are marked separately from ordinary OCR candidates and
always enter MinerU-native recognition even after the ordinary page/document
request limits are exhausted. The limits still cap non-recovery OCR review;
crop/protocol failures continue to fail closed, while the circuit breaker skips
only ordinary OCR candidates until every valid recovery crop has been sent.
Reports expose recovered candidate, response, selection, no-response, unsent,
and limit-bypass counts. `native_recovered_unsent` excludes crops that reached
vLLM but produced an invalid or length-truncated response.
Recovered boxes that receive no usable text are retained in middle JSON for
audit, marked `fusion_visualization_hidden`, and omitted from `*_span.pdf` so an
unconfirmed ink fragment is not presented as a meaningful OCR bbox. For a newly
recovered amount only, a repeated thousands/decimal separator such as
`74.791.00` is deterministically normalized to `74,791.00` after recognition.

Every bbox receives a `bbox_recognition` audit decision with OCR/VLM candidates,
unchanged bbox, field type, similarity, selected source, and reason. Batch audit
records include IDs, image count, latency, finish reason, prompt/completion token
usage, request-limit outcomes, and errors.
Each report also records machine-checked `bbox_unchanged` and
`table_structure_unchanged` invariants; either failure prevents an enablement
recommendation.
The English UI summarizes VLM selections, OCR fallbacks, and recognition errors.
It also surfaces protocol-token echoes and invalid VLM output counts so an
incompatible recognizer is visible without opening the raw fusion report.

For a separate instruction-following Vision endpoint:

```json
{
  "fusion": {
    "recognizer": {
      "enabled": true,
      "base_url": "http://vision-recognizer.example:8000",
      "model": "document-vision-model",
      "temperature": 0.0,
      "top_p": 1.0,
      "seed": 42,
      "structured_output_mode": "json_schema",
      "max_tokens": 1024,
      "max_context_tokens": 8192,
      "context_reserve_tokens": 2048,
      "max_batch_size": 8,
      "max_images_per_request": 8,
      "max_image_limit_retries": 2,
      "include_row_image": true,
      "include_column_image": false,
      "include_table_image": false
    }
  }
}
```

Keep the feature disabled until a reviewed bbox reference set proves lower CER
and no regressions. Evaluate a fusion report with:

```bash
python projects/custom_hybrid/recognition_eval.py \
  --report output/document_fusion.json \
  --reference bbox_reference.json \
  --output bbox_recognition_ab.json
```

After changing only fusion policy, replay the audited OCR/VLM candidates without
another model request by adding `--reselect-config workflow.local.json`. Replay
never changes bbox or raw candidates; it recalculates selected source, reason,
and evaluation metrics with the current recognizer guards.

Reference items use `{"page": 0, "bbox": [x0, y0, x1, y1], "text": "ground truth"}`.
The evaluator reports OCR, raw VLM, and selected exact accuracy/CER, improved and
regressed items, bbox-key coverage, geometry/Table-structure invariants, and
whether the evidence supports enabling the recognizer by default. To run the
configured recognizer on only the reviewed reference boxes before evaluating:

```bash
python projects/custom_hybrid/recognition_trial.py \
  --config workflow.local.json \
  --middle output/ocr/document_middle.json \
  --document input/document.pdf \
  --reference bbox_reference.json \
  --output bbox_recognition_trial.json
```

Default-enable evidence also requires complete operational audit: every
candidate must receive a valid response, every batch must finish with `ok`, and
there must be no request errors, invalid outputs, protocol echoes, candidate
limit, or batch-quality circuit breaker. Add
`--require-default-enable-evidence` in CI to exit with status 2 when any
accuracy, invariant, coverage, or operational gate fails.

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
6. When enabled and within its request budget, send remaining conflicts to the
   visual verifier with four possible crops:
   whole table, target row, target column, and target cell. The verifier may only
   choose `hybrid` or `pipeline`; it cannot invent or merge a third value.
7. Replace only the selected cell's inner HTML, then parse the rebuilt table and
   require its structure signature to remain identical to Hybrid. Unsafe cell
   HTML, invalid reconstruction, verifier errors, missing bboxes, and request
   limits all keep Hybrid.

The fused table copies Pipeline Cell geometry into the final Hybrid table and
synchronizes every Cell's `text` with the final fused HTML. Bounding-box PDFs draw
only OCR content-tight boxes in cyan; Cell geometry stays available in middle
JSON. If OCR confidence is absent, the safe default
`table_cell_allow_unscored_ocr=false` keeps the Hybrid candidate unless an
explicitly enabled visual verifier selects another existing candidate.

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

The verifier uses JSON mode by default and is constrained to exact existing
candidate choices. Optional page-level reconciliation is even stricter: it sees
the full page, Hybrid text, already recovered text, and a bounded manifest of OCR
candidate IDs. It may return only IDs already present in that manifest; arbitrary
transcription, new bboxes, and unknown IDs are discarded. Enable it only when a
separate instruction-following Vision endpoint is configured:

```json
{
  "fusion": {
    "reconciliation": {
      "enabled": true,
      "max_pages_per_document": 20,
      "max_candidates_per_page": 120
    },
    "verifier": {
      "enabled": true,
      "base_url": "http://vision-verifier.example:8000",
      "model": "instruction-vision-model",
      "json_mode": true
    }
  }
}
```

Keep reconciliation disabled when the only available model is the extraction-
specialized `mineru-claim-forms`; the deterministic OCR recovery and spatial
coverage audit do not depend on the verifier.

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

The Custom Hybrid task API is intentionally unauthenticated for trusted local and
LAN deployments. It can bind directly to a container or LAN interface:

```bash
python projects/custom_hybrid/api.py \
  --config workflow.local.json \
  --host 0.0.0.0 \
  --port 8010 \
  --output-root /work/mineru-output/custom-hybrid-api
```

There is no Custom Hybrid bearer-token header. Do not expose this service to the
public internet; use a trusted LAN, VPN, firewall, or authenticated reverse proxy
if the network itself is not trusted. The server uses
`workflow.local.json` as its baseline. API tasks default to the `balanced` cost
profile, which forces `effort=medium`, caps output at 2048 tokens, disables formula
and image analysis, and prevents verifier/recognizer/reconciliation requests. It
does not disable Table extraction, Pipeline OCR, deterministic fusion, or cyan
content bboxes. Submit `extraction_mode=bbox_vlm` after either cost profile to
enable the Pipeline OCR -> bbox repair -> local VLM path; this task override
enables both the required Table recognizer and the selective geometry-review /
pixel-refinement pass, while disabling unrelated verifier/reconciliation
requests. In BBox VLM
mode with a general structured VLM, `balanced` sends target and row crops while
keeping whole-Table images off; `quality` also enables the whole-Table context
image. MinerU-native recognition always sends one target crop only. Both
structured profiles enforce the eight-image request budget. Submit
`cost_profile=quality` to use the corresponding workflow configuration instead.
Each task may safely override `effort`,
parse method, language, `temperature`, `top_p`, `seed`, `max_tokens`, and
`repetition_penalty`; the API validates their types and ranges. Task generation
values are applied after prompt-specific workflow rules. In `bbox_vlm` mode,
temperature, top-p, seed, and max-token overrides are also copied into a
separately configured general bbox recognizer. MinerU-native requests deliberately
pin temperature/top-p to `0.0/0.01` and enforce their stage token cap; the task
seed remains part of the request and cache key.
Upstream URL, credentials, proxy binding,
fusion thresholds, and arbitrary vLLM arguments remain server-controlled. Tasks
run serially because each extraction owns a local parameter-proxy port and local
OCR resources. Completed ZIP files include `task_parameters.json` with the
effective task-scoped settings and `vllm_requests.jsonl` with per-request
generation audit data.

Endpoints:

- `GET /health`: service status;
- `POST /tasks`: upload PDF/images plus optional task parameters and receive a task id;
- `GET /tasks/{task_id}`: poll status;
- `GET /tasks/{task_id}/result`: download the fused ZIP;
- `GET /tasks/{task_id}/markdown`: read Markdown and content-list output;
- `GET /tasks/{task_id}/asset`: read a validated Markdown image asset;
- `GET /tasks/{task_id}/report`: read `fusion_summary.json`;
- `DELETE /tasks/{task_id}`: remove a completed/failed task and its files;
- `POST /file_parse`: wait synchronously and return the fused ZIP.

Example task-level parameter override:

```bash
curl -X POST \
  -F "files=@invoice.pdf" \
  -F "cost_profile=balanced" \
  -F "extraction_mode=bbox_vlm" \
  -F "recovery_max_tables=3" \
  -F "recovery_max_proposals=30" \
  -F "recovery_min_confidence=0.85" \
  -F "effort=medium" \
  -F "temperature=0.1" \
  -F "top_p=0.95" \
  -F "seed=123" \
  -F "max_tokens=2048" \
  http://10.100.0.30:6108/tasks
```

From a Mac, only Python and `httpx` are required. Run the client from this repo:

```bash
python -m pip install httpx

python projects/custom_hybrid/api_client.py \
  --url http://10.100.0.30:8010 \
  --input ~/Documents/invoice.pdf \
  --output ~/Documents/invoice-fused.zip \
  --cost-profile balanced \
  --extraction-mode bbox_vlm \
  --recovery-max-tables 3 \
  --recovery-max-proposals 30 \
  --recovery-min-confidence 0.85
```

The client accepts either one supported file or a directory and streams the ZIP
to disk. The Jupyter/container port must be published or reverse-proxied before a
Mac can reach it. Task state is in memory and does not survive a service restart;
task files remain under `--output-root` until deleted.

If `jupyter-server-proxy` is installed and direct port publishing is unavailable,
use `http://<jupyter-host>:<jupyter-port>/proxy/8010` as `--url` and export the
Jupyter access token through `JUPYTER_TOKEN`. The client sends that token as a
query parameter for the Jupyter proxy; the Custom Hybrid API itself remains
unauthenticated.

## 5. Run the lightweight local UI

The local UI is a small FastAPI/HTML application. It runs on the Mac and proxies
uploads, task polling, reports, and ZIP downloads to the remote Custom Hybrid
service. It does not install or run MinerU models locally.

```bash
python -m pip install fastapi uvicorn httpx python-multipart

python projects/custom_hybrid/ui.py \
  --remote-url http://10.100.0.30:6108 \
  --host 127.0.0.1 \
  --port 7860
```

Open `http://127.0.0.1:7860`. The UI follows the official MinerU workspace shape:
task controls on the left, document preview in the center, and extraction output
on the right. Result tabs provide Markdown Rendering, Markdown Text, Content List
JSON, and the Custom Hybrid Fusion Report. Relative Markdown images are served
through a task-scoped, image-only endpoint with path traversal protection.

Extraction and vLLM Generation controls are task-scoped; the Fusion Report tab
includes the accepted task parameter snapshot. The English `Extraction Mode`
selector offers `Hybrid Fusion` and `OCR + BBox Repair + VLM`. After a task
completes, the same
selected files and parameters remain available and Convert changes to Convert
Again, so rerunning does not require Clear. For PDF inputs, the fused workflow
generates `*_layout.pdf`, `*_span.pdf`, `*_forms.pdf`, and
`*_form_cells.pdf`; Document Preview automatically switches to `*_span.pdf`,
where OCR content-tight boxes are cyan. `*_forms.pdf` is a routing-only
diagnostic: purple boxes show large ruled Form/Table regions detected
independently of MinerU block types, while regions already covered by MinerU
Tables are excluded. `*_form_cells.pdf` keeps the purple outer region and adds
cyan structural leaf Cells. Blue dashed boxes appear only when handwriting or
other connected ink crosses a Cell boundary; they show the adaptive
`recognition_bbox` crop intended for VLM recognition and may overlap adjacent
Cells without changing the structural grid. Full ruled grids use physical
vertical separators; sparse forms
use long horizontal rules, aligned field underlines, and only large OCR vertical
gaps. Parent bands are stored in `form_rows`, leaf recognition targets in
`form_cells`, and neither stage rewrites layout blocks, Table HTML, or source OCR
spans. Existing Table Cell geometry remains in middle JSON and is intentionally
not drawn in `*_span.pdf`. Use Original / Bounding Boxes to switch views. The UI
can bind directly to a trusted LAN interface without an additional flag.

`span.pdf` generation and BBox crop rendering use the Pipeline-normalized
`*_origin.pdf` artifact first, so rendered pixels share the OCR coordinate system
and older PDFium builds do not need to parse unusual source PDFs. They fall back
to the uploaded task input when no normalized PDF exists. The preview endpoint
also regenerates a missing `*_span.pdf` on demand for completed tasks and validates
that the generated PDF is non-empty with the expected page count. Run `doctor`
after deployment; `pypdf` and `reportlab` are required for these visualization
artifacts and are installed by the base project dependencies.

Cell geometry is retained even when a table model supplies valid Cell bboxes but
omits logical row/column indices. Logical indices remain required for automatic
cell-text replacement, but not for cyan OCR-content rendering.
When Preview is opened, completed tasks also recover missing Cell geometry from
the sibling Pipeline OCR middle JSON before deciding whether `*_span.pdf` needs
to be redrawn.

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
