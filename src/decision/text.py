"""Ticket triage example: yes/no, single/multi choice, and score questions."""

import time

from model import load_model


def main() -> None:
    model = load_model()

    questions = {
        "refund":  {"type": "noul",   "instructions": "Is the customer asking for a refund?"},
        "team":    {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "Charges, refunds, invoices",
                                 "shipping": "Delivery and parcels",
                                 "technical": "App or site faults"}},
        "topics":  {"type": "choice", "multiple": True,
                    "instructions": "Which topics does the message raise?",
                    "criteria": {"billing": "Payments and charges",
                                 "delivery": "Shipping and parcels",
                                 "login": "Signing in to the account"}},
        "urgency": {"type": "score",  "instructions": "How urgent is this ticket?",
                    "criteria": ["low", "medium", "high"]},
    }

    tickets = [
        # the model-card example
        "I was charged twice and my parcel never arrived. Please refund one charge.",
        # negative: no refund / single topic
        "Hi, I forgot my password and can't log into my account since this morning.",
        "Please don't refund anything, I just wanted to say your parcel delivery was great!",
    ]

    for i, state in enumerate(tickets, 1):
        t0 = time.time()
        out = model.system_one(state, questions)
        a = out["answers"]
        delta = (time.time() - t0) * 1000
        urgency = a["urgency"]["legend"][str(int(a["urgency"]["score"]))]
        print(f"--- Ticket {i}  ({delta:.0f} ms)")
        print(f"refund P(yes): {a['refund']['noul']:.3f}")
        print(f"team:          {a['team']['choice']}")
        print(f"topics:        {a['topics']['choices']}")
        print(f"urgency:       {urgency}")


if __name__ == "__main__":
    main()
