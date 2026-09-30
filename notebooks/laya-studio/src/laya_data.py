"""Dataset helpers for Laya Studio (pure Python, no torch needed).

A record is one document plus the decisions labelled for it:

    {"state": {"text": "..."},                      # any text / JSON the model reads
     "questions": {"doc_type": {"type": "choice", "instructions": "...", "criteria": {...}}, ...},
     "gold": {"doc_type": "receipt", "vat_charged": true, "risk_level": 2}}

`gold` values may be a plain label (as above) or a dict with "label" and/or "probabilities",
which is the format of the official LocalLLaMA/typed-decisions dataset.
"""
import copy
import csv
import io
import json

QTYPE_NAMES = ("choice", "score", "noul")

# ---------------------------------------------------------------- starter templates
TEMPLATES = {
    "receipts_bills": {
        "title": "Bills & receipts",
        "questions": {
            "doc_type": {
                "type": "choice",
                "instructions": "What kind of financial document is this?",
                "criteria": {
                    "invoice": "a bill requesting payment, usually with an invoice number and due date",
                    "receipt": "proof that a payment was already made at a shop, restaurant or online",
                    "utility_bill": "electricity, water, internet, phone or gas bill",
                    "bank_statement": "a list of account transactions issued by a bank or wallet",
                    "payslip": "salary or wage statement",
                    "tax_document": "VAT/PAN certificate, tax return or tax receipt",
                    "other": "none of the above",
                },
            },
            "expense_category": {
                "type": "choice",
                "instructions": "Which expense category does this document belong to?",
                "criteria": {
                    "food_dining": "restaurants, cafes, food delivery",
                    "groceries": "supermarkets, grocery and household goods",
                    "transport": "fuel, taxi, ride-hailing, bus, parking, vehicle service",
                    "utilities": "electricity, water, internet, phone",
                    "rent": "office or home rent, lease",
                    "office_supplies": "stationery, equipment, printing",
                    "software": "software, SaaS, cloud or domain subscriptions",
                    "travel": "flights, hotels, tours",
                    "medical": "hospital, clinic, pharmacy",
                    "professional_services": "legal, accounting, consulting, contractors",
                    "other": "none of the above",
                },
            },
            "payment_method": {
                "type": "choice",
                "instructions": "How was this paid, or how is it to be paid?",
                "criteria": {
                    "cash": "cash payment",
                    "card": "debit or credit card",
                    "bank_transfer": "bank transfer, Fonepay/QR to a bank account, or deposit",
                    "mobile_wallet": "eSewa, Khalti, IME Pay or another wallet",
                    "cheque": "cheque",
                    "unknown": "not stated",
                },
            },
            "vat_charged": {
                "type": "noul",
                "instructions": "Does the document show VAT or another tax being charged?",
            },
            "needs_review": {
                "type": "noul",
                "instructions": "Is information missing, unreadable, duplicated or inconsistent "
                                "(for example totals that do not add up) so a person should review it?",
            },
        },
    },
    "fintech_transactions": {
        "title": "Fintech transactions",
        "questions": {
            "txn_type": {
                "type": "choice",
                "instructions": "What type of transaction is this?",
                "criteria": {
                    "payment": "payment to a merchant or biller",
                    "refund": "money returned for an earlier payment",
                    "transfer": "person-to-person or account-to-account transfer",
                    "withdrawal": "cash withdrawal or cash-out",
                    "deposit": "cash-in, top-up or deposit",
                    "fee": "service charge, commission or penalty",
                    "salary": "salary or payroll credit",
                    "loan_repayment": "loan or EMI repayment",
                    "other": "none of the above",
                },
            },
            "is_suspicious": {
                "type": "noul",
                "instructions": "Does this transaction look suspicious or potentially fraudulent?",
            },
            "risk_level": {
                "type": "score",
                "instructions": "How risky is this transaction?",
                "criteria": [
                    "low: routine, consistent with normal activity",
                    "moderate: slightly unusual amount, time or counterparty",
                    "elevated: several unusual signals, should be checked",
                    "high: strong fraud or money-laundering signals, block or escalate",
                ],
            },
        },
    },
    "fintech_support": {
        "title": "Fintech customer support",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "What does the customer want?",
                "criteria": {
                    "failed_payment": "a payment failed or is stuck/pending",
                    "duplicate_charge": "charged twice for the same thing",
                    "refund_request": "wants money back",
                    "account_access": "cannot log in, locked account, PIN or OTP problem",
                    "kyc_verification": "KYC, identity documents or verification",
                    "balance_or_statement": "asks about balance, statement or transaction history",
                    "card_issue": "card blocked, lost, declined or not received",
                    "other": "none of the above",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this request?",
                "criteria": [
                    "low: general question, no money at risk",
                    "normal: needs an answer today",
                    "high: money is stuck or the customer cannot transact",
                    "critical: money lost, fraud or account compromised right now",
                ],
            },
            "needs_human": {
                "type": "noul",
                "instructions": "Does this need a human agent rather than an automated reply?",
            },
        },
    },
}

_TRUE = {"true", "yes", "y", "1", "t", "ho", "हो"}
_FALSE = {"false", "no", "n", "0", "f", "haina", "होइन"}


# ---------------------------------------------------------------- questions
def normalize_question(qid, q):
    """Validate one question definition and return a clean copy."""
    if not isinstance(q, dict):
        raise ValueError("question %r must be an object" % qid)
    t = q.get("type")
    if t not in QTYPE_NAMES:
        raise ValueError("question %r: type must be one of %s" % (qid, ", ".join(QTYPE_NAMES)))
    out = {"type": t, "instructions": str(q.get("instructions") or qid)}
    crit = q.get("criteria")
    if t == "choice":
        if isinstance(crit, list):
            crit = {str(k): "" for k in crit}
        if not isinstance(crit, dict) or len(crit) < 2:
            raise ValueError("question %r: a choice needs criteria with at least 2 options" % qid)
        out["criteria"] = {str(k): ("" if v is None else v) for k, v in crit.items()}
    elif t == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise ValueError("question %r: a score needs criteria as a list of at least 2 levels" % qid)
        out["criteria"] = [str(c) for c in crit]
    else:
        if crit:
            if not isinstance(crit, dict) or not set(crit) <= {"true", "false"}:
                raise ValueError("question %r: noul criteria may only have 'true'/'false' keys" % qid)
            out["criteria"] = dict(crit)
    return out


def option_keys(q):
    """Option names in the order Laya renders them (noul is always [false, true])."""
    t = q["type"]
    if t == "choice":
        return list(q["criteria"].keys())
    if t == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["false", "true"]


def label_index(q, value):
    """Turn a human label into an option index. Raises ValueError when it doesn't fit."""
    t = q["type"]
    if isinstance(value, str):
        value = value.strip()
    if t == "noul":
        if isinstance(value, bool):
            return int(value)
        s = str(value).strip().lower()
        if s in _TRUE:
            return 1
        if s in _FALSE:
            return 0
        raise ValueError("expected yes/no (true/false), got %r" % (value,))
    if t == "choice":
        keys = option_keys(q)
        s = str(value)
        if s in keys:
            return keys.index(s)
        low = [k.lower() for k in keys]
        if s.lower() in low:
            return low.index(s.lower())
        raise ValueError("%r is not one of: %s" % (value, ", ".join(keys)))
    # score
    n = len(q["criteria"])
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        i = int(round(value))
    else:
        s = str(value).strip()
        try:
            i = int(round(float(s)))
        except ValueError:
            names = [str(c).split(":")[0].strip().lower() for c in q["criteria"]]
            if s.lower() in names:
                i = names.index(s.lower())
            else:
                raise ValueError("%r is not a level number 0-%d or one of: %s" % (value, n - 1, ", ".join(names)))
    if not 0 <= i < n:
        raise ValueError("level %d is outside 0-%d" % (i, n - 1))
    return i


def gold_to_target(q, g):
    """Return (target distribution over options, label index) for one gold answer."""
    keys = option_keys(q)
    k = len(keys)
    probs = None
    label = g
    if isinstance(g, dict):
        probs = g.get("probabilities")
        label = g.get("label", g.get("choice", g.get("score", g.get("noul"))))
    if isinstance(probs, dict) and probs:
        target = [max(0.0, float(probs.get(key, 0.0))) for key in keys]
    elif q["type"] == "noul" and isinstance(label, float) and 0.0 < label < 1.0:
        target = [1.0 - label, label]          # soft yes/no probability
    else:
        if label is None:
            raise ValueError("no label given")
        target = [0.0] * k
        target[label_index(q, label)] = 1.0
    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / k] * k
    return target, max(range(k), key=lambda i: target[i])


# ---------------------------------------------------------------- records
def normalize_record(rec):
    """Validate a record; returns a clean copy (only questions that have a gold answer)."""
    if not isinstance(rec, dict):
        raise ValueError("record must be a JSON object")
    state, questions, gold = rec.get("state"), rec.get("questions"), rec.get("gold")
    # The official dataset stores these three as JSON strings.
    if isinstance(state, str) and state.strip()[:1] in "{[":
        try:
            state = json.loads(state)
        except ValueError:
            pass
    if isinstance(questions, str):
        questions = json.loads(questions)
    if isinstance(gold, str):
        gold = json.loads(gold)
    if state in (None, "", {}, []):
        raise ValueError("missing 'state' (the document text or JSON)")
    if not isinstance(questions, dict) or not isinstance(gold, dict):
        raise ValueError("'questions' and 'gold' must be objects")
    out_q, out_g = {}, {}
    for qid, g in gold.items():
        if qid not in questions:
            continue
        q = normalize_question(qid, questions[qid])
        try:
            gold_to_target(q, g)
        except ValueError as e:
            raise ValueError("%s: %s" % (qid, e))
        out_q[qid], out_g[qid] = q, g
    if not out_g:
        raise ValueError("no labelled answers (gold) that match a question")
    return {"state": state, "questions": out_q, "gold": out_g}


def parse_jsonl(text):
    records, errors = [], []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(normalize_record(json.loads(line)))
        except (ValueError, TypeError) as e:
            errors.append("line %d: %s" % (n, e))
    return records, errors


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        records, errors = parse_jsonl(f.read())
    return records


def parse_csv(text, questions):
    """CSV import. Columns named after question ids hold labels; every other column is the document.

    A single 'text' column becomes {"text": ...}; several columns become a JSON state such as
    {"merchant": ..., "amount": ..., "text": ...}. Empty label cells are skipped.
    """
    questions = {qid: normalize_question(qid, q) for qid, q in questions.items()}
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if not reader.fieldnames:
        return [], ["CSV has no header row"]
    fields = [f.strip() for f in reader.fieldnames if f is not None]
    qcols = [f for f in fields if f in questions]
    scols = [f for f in fields if f not in questions]
    if not qcols:
        return [], ["no column matches a question id of this template (%s)" % ", ".join(questions)]
    if not scols:
        return [], ["no document column found - add a 'text' column"]
    records, errors = [], []
    for n, row in enumerate(reader, 2):
        row = {(k or "").strip(): (v or "").strip() for k, v in row.items() if k is not None}
        state = {c: row.get(c, "") for c in scols if row.get(c, "")}
        if not state:
            errors.append("row %d: empty document" % n)
            continue
        gold, qs, bad = {}, {}, None
        for c in qcols:
            v = row.get(c, "")
            if v == "":
                continue
            try:
                label_index(questions[c], v)
            except ValueError as e:
                bad = "row %d, %s: %s" % (n, c, e)
                break
            gold[c], qs[c] = v, questions[c]
        if bad:
            errors.append(bad)
            continue
        if not gold:
            errors.append("row %d: no labels" % n)
            continue
        records.append({"state": state, "questions": qs, "gold": gold})
    return records, errors


def csv_example(questions):
    """A tiny example CSV for the help text."""
    qids = list(questions)
    row = []
    for qid in qids:
        q = normalize_question(qid, questions[qid])
        row.append(option_keys(q)[0] if q["type"] == "choice" else "yes" if q["type"] == "noul" else "1")
    return "text," + ",".join(qids) + "\n\"<document text>\"," + ",".join(row) + "\n"


def state_preview(state, n=160):
    s = state if isinstance(state, str) else (state.get("text") if isinstance(state, dict) and
                                               isinstance(state.get("text"), str) else json.dumps(state, ensure_ascii=False))
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def gold_display(q, g):
    target, idx = gold_to_target(q, g)
    if q["type"] == "choice":
        return option_keys(q)[idx]
    if q["type"] == "noul":
        return "yes" if idx == 1 else "no"
    return "level %d" % idx


def dataset_stats(records):
    per_q = {}
    for rec in records:
        for qid, g in rec["gold"].items():
            q = rec["questions"][qid]
            d = per_q.setdefault(qid, {"type": q["type"], "counts": {}})
            try:
                lab = gold_display(q, g)
            except ValueError:
                continue
            d["counts"][lab] = d["counts"].get(lab, 0) + 1
    return {"records": len(records),
            "decisions": sum(len(r["gold"]) for r in records),
            "questions": per_q}


def default_templates():
    return copy.deepcopy(TEMPLATES)


def validate_templates(t):
    if not isinstance(t, dict) or not t:
        raise ValueError("templates must be a non-empty object")
    out = {}
    for tid, tpl in t.items():
        if not isinstance(tpl, dict) or not isinstance(tpl.get("questions"), dict) or not tpl["questions"]:
            raise ValueError("template %r needs a 'questions' object" % tid)
        out[str(tid)] = {"title": str(tpl.get("title") or tid),
                         "questions": {qid: normalize_question(qid, q) for qid, q in tpl["questions"].items()}}
    return out
