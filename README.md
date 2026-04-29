# LLM Reliability & Hallucination Evaluation for Legal Drafting

This repository now includes a practical evaluation harness for your PhD comparative study on reducing hallucinations in legal contract drafting.

## What is implemented

- **Comparative baselines** across providers:
  - OpenAI (`gpt-4.1-mini`) as baseline.
  - Gemini (`gemini-2.5-flash`) as baseline.
  - Groq-hosted model (`llama-3.3-70b-versatile`) as baseline.
- **Two evaluation modes** per model:
  - `naive`: prompt-only (no source context).
  - `grounded`: prompt + source contract text.
- **Dataset-driven runs** using JSONL inputs.
- **Automatic metrics**:
  - Unsupported claim rate (proxy hallucination metric).
  - Expected point coverage.
  - Risk flag (`low`/`medium`/`high`).
- **CSV outputs** for thesis tables and plotting.

## Project structure

- `main.py` — entrypoint.
- `study/config.py` — runtime settings from environment variables.
- `study/dataset.py` — JSONL loading.
- `study/providers.py` — OpenAI, Gemini, and Groq provider clients.
- `study/scoring.py` — scoring logic.
- `study/evaluate.py` — orchestration and summary output.
- `data/legal_contract_eval.jsonl` — starter legal evaluation dataset.
- `prompts/legal_rubric.md` — human legal review rubric.

## Quick start

1. Set API keys:

```bash
export GEMINI_API_KEY=...
export GROQ_API_KEY=...
# Optional:
# export OPENAI_API_KEY=...
```

2. Run evaluation:

```bash
python main.py
```

3. Results are written to:

- `results/eval_results.csv`

## Environment variables

- `EVAL_DATASET_PATH` (default: `data/legal_contract_eval.jsonl`)
- `EVAL_OUTPUT_PATH` (default: `results/eval_results.csv`)
- `EVAL_TEMPERATURE` (default: `0.0`)
- `EVAL_MAX_TOKENS` (default: `800`)
- `EVAL_PROVIDERS` (default: `gemini,groq`; options: `gemini,groq,openai`)

## Next extensions for thesis-grade rigor

- Add adjudicated human legal scoring with two raters.
- Expand dataset to 100+ examples across NDA/MSA/SOW/DPA.
- Add citation verification metric and contradiction checks.
- Add significance testing and confidence intervals in reporting.


## Troubleshooting

- If you see NumPy/Pandas native library errors (for example `libstdc++.so.6` missing), this harness no longer depends on pandas for evaluation output.
- Run `python main.py` after installing only pure-Python requirements plus `requests`.


## Visualizing results (for thesis presentation)

After running `python main.py`, generate an easy-to-understand chart:

```bash
python -m study.visualize
```

This creates:
- `results/eval_dashboard.png`

The dashboard includes two side-by-side bar charts:
- **Unsupported Claim Rate** (lower is better)
- **Expected Coverage** (higher is better)
