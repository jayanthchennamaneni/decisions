"""Vision example: counting, boxes, and reading values from text and images."""

import json
import time

from PIL import Image, ImageDraw, ImageFont
from transformers.image_utils import load_image

try:
    from .model import load_model
except ImportError:  # running as a plain script, not as a package module
    from model import load_model


def timed(model, label, state, questions, media=None):
    t0 = time.time()
    out = model.system_one(state, questions, media=media)
    dt = (time.time() - t0) * 1000
    print(f"--- {label}  ({dt:.0f} ms)")
    return out["answers"]


def main() -> None:
    model = load_model()

    # ---- Part A: COCO classic image (ground truth: 2 cats, 2 remotes, no people/dogs)
    image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
    a = timed(model, "A. cats + boxes", None, {
        "cats":   {"type": "number", "instructions": "How many cats are there?", "range": [0, 10], "step": 1},
        "dogs":   {"type": "number", "instructions": "How many dogs are there?", "range": [0, 10], "step": 1},
        "remote": {"type": "box",    "instructions": "Where is the remote control on the left?"},
        "every":  {"type": "boxes",  "instructions": "Every cat.", "max": 4},
        "person": {"type": "noul",   "instructions": "Is there a person in the picture?"},
    }, media=[image])

    print("cats:", a["cats"]["number"], "| dogs:", a["dogs"]["number"], "| person P(yes):", round(a["person"]["noul"], 3))
    print("remote box:", {k: round(v, 2) for k, v in a["remote"]["box"].items()}, "| found:", round(a["remote"]["found"], 2))
    print("every-cat boxes count:", a["every"]["count"])

    # ---- Part B: read values from TEXT (docs example; ground truth: VAT is absent)
    invoice = "ACME SA\nInvoice 2026-117\nTotal due: 1 234,50 EUR\nPay before 03/04/2026 (day/month/year)"
    b = timed(model, "B. read from text", invoice, {
        "payee": {"type": "text",   "instructions": "Who issued the invoice?"},
        "total": {"type": "number", "instructions": "How much is due?"},
        "due":   {"type": "date",   "instructions": "By when must it be paid?"},
        "vat":   {"type": "number", "instructions": "How much VAT is charged?"},
    })
    for q in ("payee", "total", "due", "vat"):
        r = b[q]
        print(f"{q}: value={r.get('text', r.get('number', r.get('date')))}  found={round(r['found'], 2)}  span={r.get('span')}")

    # ---- Part C: read values from a SYNTHETIC document image (VAT deliberately absent)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 64)
    except Exception:
        font = ImageFont.load_default(size=64)

    doc = Image.new("RGB", (1152, 768), "white")
    draw = ImageDraw.Draw(doc)
    draw.text((90, 80),  "ACME SA", fill="black", font=font)
    draw.text((90, 260), "Invoice A-2041", fill="black", font=font)
    draw.text((90, 440), "Total due: 129,90 EUR", fill="black", font=font)
    draw.text((90, 600), "Pay before 30/11/2026", fill="black", font=font)
    doc.save("invoice.png")   # open it — if YOU can't read it, the test is unfair
    print("Check invoice.png legibility on your screen")

    c = timed(model, "C. read from image", None, {
        "invoice_no": {"type": "text",   "instructions": "What is the invoice number?"},
        "total":      {"type": "number", "instructions": "How much is due?"},
        "due":        {"type": "date",   "instructions": "By when must it be paid?"},
        "vat":        {"type": "number", "instructions": "How much VAT is charged?"},
    }, media=[doc])
    for q, key in (("invoice_no", "text"), ("total", "number"), ("due", "date"), ("vat", "number")):
        r = c[q]
        print(f"{q}: value={r[key]}  found={round(r['found'], 2)}  span={r.get('span')}")

    # ---- Part D: does a text hint change how well the image is read?
    probes = {
        "sees_amount":  {"type": "noul",   "instructions": "Does the image contain the amount 129,90 EUR?"},
        "pick_amount":  {"type": "choice", "instructions": "Which amount does the document show?",
                         "criteria": {"a": "1 290,00", "b": "129,90", "c": "12 990"}},
        "pick_word":    {"type": "choice", "instructions": "Which word appears in the image?",
                         "criteria": {"acme": "ACME", "zeta": "ZETA", "omega": "OMEGA"}},
        "read_total":   {"type": "number", "instructions": "How much is due?"},
        "read_invoice": {"type": "text",   "instructions": "What is the invoice number?"},
    }

    d_results = {
        "D1 image only":        timed(model, "D1 image only", None, probes, media=[doc]),
        "D2 text hint + image": timed(model, "D2 text hint + image", "An invoice document from ACME SA.", probes, media=[doc]),
    }

    for name, answers in d_results.items():
        print(f"===== {name}")
        print("sees_amount P(yes):", round(answers["sees_amount"]["noul"], 3))
        print("pick_amount:", answers["pick_amount"]["choice"], "| probs:",
              {k: round(v, 2) for k, v in answers["pick_amount"]["probabilities"].items()})
        print("pick_word:", answers["pick_word"]["choice"], "| probs:",
              {k: round(v, 2) for k, v in answers["pick_word"]["probabilities"].items()})
        print("read_total:", json.dumps(answers["read_total"]))
        print("read_invoice:", json.dumps(answers["read_invoice"]))


if __name__ == "__main__":
    main()
