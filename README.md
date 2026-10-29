# decision

Minimal reference implementation of **structured decision queries** against
[LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B) — a small (2B) vision-language
model that answers typed questions (probabilities, choices, scores, numbers,
bounding boxes, dates) instead of free-form text.

## Examples

| Script     | Demonstrates                                                                 |
|------------|------------------------------------------------------------------------------|
| `text.py`  | Ticket triage: yes/no probability, single choice, multi-select, urgency score |
| `vision.py`| Counting, bounding boxes, and reading fields (text, number, date) from raw text *and* document images; includes a text-hint ablation |

## Setup

Requires Python 3.12+ and an Apple Silicon Mac (MPS) or NVIDIA GPU (CUDA).

```bash
uv sync
```

## Run

```bash
uv run python src/decision/text.py
uv run python src/decision/vision.py
```

These are plain scripts, not package entry points — just run the file.
`vision.py` also writes an `invoice.png` synthetic document next to the project
root so you can eyeball what the model was shown.

## How it works

Every query is a dictionary of typed questions passed to `model.system_one(state, questions)`:

- `state` — the input text (or `None` for image-only queries)
- `questions` — keyed by answer slot, each with a `type` and `instructions`
  - `noul` — yes/no, returns a probability
  - `choice` — pick from `criteria` labels (supports `multiple: true`)
  - `score` — ordered scale, e.g. `["low", "medium", "high"]`
  - `number`, `box`, `boxes`, `text`, `date` — extraction types

Answers come back in the same keys with type-specific payloads (`noul`,
`choice`, `number`, `box`, `found` confidence, `span`, …).

## Project layout

```
src/decision/
├── model.py    # shared d1-2B loader with mps/cuda/cpu fallback
├── text.py     # text-only example
└── vision.py   # image + document-reading example
```

The examples import `model.py` as a sibling module (`from model import
load_model`), so they run as plain scripts.
