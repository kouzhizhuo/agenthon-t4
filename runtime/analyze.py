#!/usr/bin/env python3
"""Offline, evidence-aware Track 4 trial; standard library only."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import unicodedata

TOKEN = re.compile(r"[a-z0-9]+")
DOC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
STOP = set("the and for from with using only each table corpus frozen predict prediction month year its as of in a to per percent latest given is this by first".split())


def tokens(text):
    return [t for t in TOKEN.findall(str(text).lower()) if len(t) > 1 and t not in STOP]


def document_text(doc):
    if isinstance(doc.get("text"), str):
        return doc["text"]
    return " ".join(s.get("text", "") for s in doc.get("spans", []) if isinstance(s, dict))


def iso_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("A canonical YYYY-MM-DD date is required")
    return date.fromisoformat(value)


@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str
    doc_date: date
    title: str
    digest: str


@dataclass(frozen=True)
class Span:
    doc_id: str
    start: int
    end: int
    text: str


def load_corpus(corpus_dir, cutoff):
    """Only manifest-declared, no-follow, digest-verified, eligible documents."""
    corpus_dir = Path(corpus_dir)
    manifest_path = corpus_dir.parent / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = corpus_dir / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Manifest may not be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs, excluded = [], []
    seen = set()
    for entry in manifest["files"]:
        if entry.get("role") != "corpus" or entry.get("path") == "corpus/manifest.json":
            continue
        parts = entry["path"].split("/")
        if len(parts) != 2 or parts[0] != "corpus" or not parts[1].endswith(".json"):
            raise ValueError("Corpus path must be corpus/<doc_id>.json")
        did = parts[1][:-5]
        if not DOC_ID.fullmatch(did) or unicodedata.normalize("NFC", did) != did or did in seen:
            raise ValueError("Invalid or duplicate manifest document id")
        seen.add(did)
        path = corpus_dir / parts[1]
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            import stat
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("Corpus document must be a regular file")
            with os.fdopen(fd, "rb", closefd=False) as f:
                raw = f.read(8 * 1024 * 1024 + 1)
        finally:
            os.close(fd)
        if len(raw) > 8 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
            raise ValueError("Corpus digest or size violation")
        obj = json.loads(raw)
        try:
            dated = iso_date(obj.get("doc_date"))
        except ValueError:
            excluded.append({"doc_id": did, "reason": "undated_or_malformed_date"})
            continue
        text = document_text(obj)
        if dated > cutoff or not text.strip():
            excluded.append({"doc_id": did, "reason": "post_cutoff" if dated > cutoff else "empty_scorer_text"})
            continue
        docs.append(Doc(did, text, dated, obj.get("title", ""), entry["sha256"]))
    return docs, excluded


def compile_spec(task):
    target = task.get("target", {})
    kind = task.get("target_type") or target.get("type")
    if kind not in ("classification", "regression", "ranking"):
        raise ValueError("Unknown target type is a specification hole; cannot invent it")
    ids = [e["entity_id"] for e in task["entities"]]
    if not ids or any(not isinstance(eid, str) or not eid or unicodedata.normalize("NFC", eid) != eid for eid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Task roster must be nonempty and unique")
    labels = target.get("labels", [])
    if kind == "classification" and not labels:
        raise ValueError("Classification label vocabulary is missing")
    known = [
        ("task_id", task["task_id"], "task.json:task_id"),
        ("target_type", kind, "task.json:target_type or target.type"),
        ("entity_roster", ids, "task.json:entities[].entity_id"),
        ("cutoff", task["cutoff_date"], "task.json:cutoff_date"),
        ("interval_level", task.get("interval_level", 0.9), "task.json:interval_level; published analysis schema"),
        ("labels", labels, "task.json:target.labels"),
        ("finite_ordered_interval", True, "qfbench2_track_analysis/alignment.py"),
        ("resolved_nonempty_citations", True, "qfbench2_track_analysis/corpus.py; shared analysis schema"),
    ]
    return {
        "task_id": task["task_id"], "target_type": kind, "labels": labels,
        "entity_roster": ids, "cutoff": task["cutoff_date"],
        "interval_level": task.get("interval_level", 0.9),
        "known_constraints": [{"name": k, "value": v, "provenance": p} for k, v, p in known],
        "holes": [
            {"name": "future_prediction_support", "status": "HOLE", "reason": "A historical citation need not entail a future prediction; production NLI has not run."},
            {"name": "predictive_quality", "status": "HOLE", "reason": "Public practice units have no resolved outcomes."},
            {"name": "future_interval_calibration", "status": "HOLE", "reason": "No labeled calibration set; historical error bands are provisional."},
        ],
    }


def binding(entity, doc):
    """Evidence of entity identity, separate from target relevance and entailment."""
    prefix = (doc.title + " " + doc.text[:7000]).lower()
    cik = str(entity.get("cik", "")).zfill(10) if entity.get("cik") else ""
    if cik and cik in doc.doc_id:
        return 6.0, "cik_in_manifest_id"
    for key in ("series_fred", "series_id", "entity_id"):
        value = str(entity.get(key, ""))
        if value and re.search(r"(?<![a-z0-9])" + re.escape(value.lower()) + r"(?![a-z0-9])", prefix):
            return 4.0, key + "_in_document"
    name = str(entity.get("name") or entity.get("series_name") or "")
    clean = re.sub(r"\b(inc|corporation|corp|company|co)\b\.?", "", name.lower()).strip(" ,.")
    if len(clean) > 3 and clean in prefix:
        return 4.0, "entity_name_in_document"
    tenor = str(entity.get("tenor", ""))
    if tenor and tenor.lower() in (doc.title + " " + doc.text[:160]).lower():
        return 5.0, "tenor_in_document_title"
    words = set(tokens(name))
    overlap = len(words.intersection(tokens(doc.title))) / max(1, len(words))
    return (2.0 * overlap, "name_token_overlap" if overlap else "no_entity_binding")


class Retriever:
    def __init__(self, docs):
        self.docs = {d.doc_id: d for d in docs}
        self.spans = []
        self.counts = []
        self.df = Counter()
        for doc in docs:
            for start in range(0, len(doc.text), 700):
                end = min(len(doc.text), start + 1000)
                span = Span(doc.doc_id, start, end, doc.text[start:end])
                counts = Counter(tokens(span.text))
                self.spans.append(span)
                self.counts.append(counts)
                self.df.update(counts.keys())
        self.avg = sum(sum(c.values()) for c in self.counts) / max(1, len(self.counts))

    def retrieve(self, entity, task, k=3):
        query = " ".join(str(entity.get(key, "")) for key in ("entity_id", "name", "series_name", "series_id", "series_fred", "tenor"))
        query += " " + task.get("target", {}).get("name", "").replace("_", " ")
        query += " " + task.get("prompt", "")[:400]
        q = Counter(tokens(query))
        bound = {did: binding(entity, doc) for did, doc in self.docs.items()}
        # Known identity bindings constrain retrieval; absence becomes an explicit hole.
        best_binding = max((v[0] for v in bound.values()), default=0)
        scored = []
        for i, (span, counts) in enumerate(zip(self.spans, self.counts)):
            b, reason = bound[span.doc_id]
            if best_binding >= 4 and b < best_binding:
                continue
            length = sum(counts.values())
            score = b * 3
            for word in q:
                tf = counts[word]
                if tf:
                    idf = math.log(1 + (len(self.spans) - self.df[word] + .5) / (self.df[word] + .5))
                    score += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * length / max(1, self.avg)))
            scored.append((score, -i, span, reason))
        scored.sort(reverse=True, key=lambda v: (v[0], v[1]))
        selected = []
        for score, _, span, reason in scored:
            if any(s.doc_id == span.doc_id and min(s.end, span.end) > max(s.start, span.start) for s, _, _ in selected):
                continue
            selected.append((span, score, reason))
            if len(selected) == k:
                break
        return selected


def numeric(value):
    try:
        result = float(str(value).replace(",", "").replace("%", "").replace("$", ""))
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def tables(text):
    lines = text.splitlines(keepends=True)
    offset, header, start, rows = 0, None, None, []
    for line in lines:
        cells = [c.strip() for c in line.strip().split("|")]
        if "|" in line:
            if header is None:
                header, start, rows = cells, offset, []
            elif len(cells) == len(header):
                rows.append(cells)
        elif header is not None:
            if rows:
                yield header, rows, start, offset
            header, start, rows = None, None, []
        offset += len(line)
    if header is not None and rows:
        yield header, rows, start, offset


def historical_signal(entity, task, docs):
    target_name = task.get("target", {}).get("name", "").lower()
    ordered = sorted(docs, key=lambda d: (-binding(entity, d)[0], d.doc_id))
    for doc in ordered:
        b, why = binding(entity, doc)
        for header, rows, start, end in tables(doc.text):
            column = None
            for i, h in enumerate(header):
                if h.lower().replace("-", "_") == target_name or h.lower().replace("-", "_") in target_name:
                    if h.lower() not in ("month", "date", "term"):
                        column = i
            # Names bind cross-sectional wide-table columns without a family identifier.
            name = str(entity.get("name") or entity.get("series_name") or "").lower()
            if column is None:
                column = next((i for i, h in enumerate(header) if h.lower() == name), None)
            if column is not None and (b >= 4 or header[column].lower() == name):
                vals = [numeric(row[column]) for row in rows]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 3:
                    return {"values": vals, "doc_id": doc.doc_id, "start": start, "end": end, "column": header[column], "binding": why, "mode": "historical_level"}
            if "revision" in target_name and b >= 4 and any("as_of" in h for h in header):
                differences = []
                for row in rows:
                    vals = [numeric(v) for v in row[1:]]
                    vals = [v for v in vals if v is not None]
                    differences += [y - x for x, y in zip(vals, vals[1:]) if y != x]
                if differences:
                    return {"values": differences, "doc_id": doc.doc_id, "start": start, "end": end, "column": "pre-cutoff vintage revisions", "binding": why, "mode": "historical_change"}
    return None


def neutral_label(labels):
    for preferred in ("inline", "flat", "no_event", "neutral", "unchanged"):
        if preferred in labels:
            return preferred
    return sorted(labels)[0]


def distress_signal(entity, docs):
    """Conditional default clauses are not evidence of an issuer's actual distress."""
    matches = []
    for doc in docs:
        if binding(entity, doc)[0] < 4:
            continue
        for match in re.finditer(r"substantial doubt|(?:we|the company) (?:has |have )?(?:failed to pay|defaulted)", doc.text, re.I):
            start, end = max(0, match.start() - 250), min(len(doc.text), match.end() + 300)
            context = doc.text[start:end].lower()
            ambiguous = any(cue in context for cue in ("alleviat", "initially raised", "evaluated whether", "would ", "could "))
            current = any(cue in context for cue in ("has concluded", "we have concluded", "there is substantial doubt", "failed to pay", "defaulted"))
            matches.append((2 if current and not ambiguous else 1, doc, start, end))
    if not matches:
        return .15, None, "No explicit issuer distress disclosure identified; generic conditional default clauses ignored."
    priority, doc, start, end = max(matches, key=lambda x: (x[0], -x[2]))
    return (.65 if priority == 2 else .5), (doc, start, end), ("Current distress disclosure prior; not a calibrated event probability." if priority == 2 else "Distress mention may be conditional or mitigated; unresolved semantic ambiguity.")


def predict(entity, task, spec, docs, retrieved):
    name = task.get("target", {}).get("name", "").lower()
    labels = spec["labels"]
    holes = []
    history = historical_signal(entity, task, docs)
    distress = None
    point, width, method = 0.0, 1.0, "zero_change_prior"
    if history:
        vals = history["values"]
        tail = vals[-min(6, len(vals)):]
        estimate = statistics.mean(tail)
        width = max(1e-6, 1.645 * statistics.stdev(vals))
        if history["mode"] == "historical_change":
            baseline = numeric(entity.get("latest_precutoff_estimate")) or 0.0
            point = baseline + estimate
            method = "latest_level_plus_mean_nonzero_historical_revision"
        else:
            point = estimate
            method = "recent_historical_mean"
    elif "trailing_4wk_net_change_pct_oi" in entity:
        point = float(entity["trailing_4wk_net_change_pct_oi"])
        width = max(5.0, abs(point))
        method = "trailing_change_persistence"
    elif "consensus_eps" in entity:
        point = float(entity["consensus_eps"])
        width = max(.5, abs(point) * .35)
        method = "consensus_prior"
    elif "prior_year_q_eps" in entity:
        point = 0.0 if "growth" in name else float(entity["prior_year_q_eps"])
        width = 50.0 if "growth" in name else max(.5, abs(point) * .5)
        method = "zero_growth_or_prior_year_eps_prior"
    elif "latest_precutoff_estimate" in entity:
        point = float(entity["latest_precutoff_estimate"])
        width = max(1.0, abs(point) * .02)
        method = "latest_published_level_prior"
    elif "start_yield_pct" in entity:
        width, method = 75.0, "zero_yield_change_prior_bps"
    elif "flat_threshold_abn_pct" in entity:
        width, method = 10.0, "zero_abnormal_return_prior_pct"
    elif "credit_event" in labels:
        point, distress, explanation = distress_signal(entity, docs)
        holes.append(explanation)
        width, method = .5, "disclosed_distress_cues_uncalibrated_prior"
    else:
        holes.append("No trustworthy target-compatible numeric signal found; zero prior used.")
    label = None
    if spec["target_type"] == "classification":
        label = neutral_label(labels)
        if "credit_event" in labels:
            label = "credit_event" if point >= .5 else "no_event"
        elif set(labels) == {"up", "down"}:
            baseline = numeric(entity.get("latest_precutoff_estimate", entity.get("prior_year_q_eps")))
            if baseline is not None and point != baseline:
                label = "up" if point > baseline else "down"
            else:
                holes.append("No direction evidence; deterministic vocabulary fallback is an unresolved prediction.")
    lo, hi = point - width, point + width
    if "credit_event" in labels:
        lo, hi = max(0, lo), min(1, hi)
    claims = []
    if distress:
        doc, start, end = distress
        claims.append({"doc_id": doc.doc_id, "span_start": start, "span_end": end,
                       "claim": "This issuer-bound passage discusses going-concern or payment distress. The event probability is a provisional prior; mitigation and future entailment remain unresolved."})
    if history:
        doc = next(d for d in docs if d.doc_id == history["doc_id"])
        start, end = history["start"], history["end"]
        claims.append({"doc_id": doc.doc_id, "span_start": start, "span_end": end,
                       "claim": f"Pre-cutoff table provides {history['column']} history. The forecast extrapolates that history; future support remains unverified."})
    for span, _, _ in retrieved[:2]:
        claims.append({"doc_id": span.doc_id, "span_start": span.start, "span_end": span.end,
                       "claim": "Pre-cutoff evidence excerpt: " + span.text[:240] + " Forecast entailment has not been checked."})
    if not claims:
        raise ValueError("No eligible corpus evidence exists; refusing to invent a citation")
    row = {"entity_id": entity["entity_id"], "point_forecast": round(point, 8),
           "interval": {"level": spec["interval_level"], "lo": round(lo, 8), "hi": round(hi, 8)}, "claims": claims}
    if label is not None:
        row["label"] = label
    diagnostic = {"entity_id": entity["entity_id"], "method": method, "holes": holes,
                  "entity_bindings": [{"doc_id": s.doc_id, "binding": why, "bm25_score": round(score, 5)} for s, score, why in retrieved],
                  "history_count": len(history["values"]) if history else 0, "interval_calibrated": False,
                  "production_nli": "NOT_RUN", "prediction_entailment": "UNKNOWN"}
    return row, diagnostic, history


def check_answer(answer, spec, docs):
    """Our executable structural checks; deliberately does not judge future semantics."""
    errors = []
    by_doc = {d.doc_id: d for d in docs}
    rows = answer.get("entity_predictions", [])
    ids = [r.get("entity_id") for r in rows]
    if len(ids) != len(spec["entity_roster"]) or len(set(ids)) != len(ids) or set(ids) != set(spec["entity_roster"]):
        errors.append("Exact unique roster set mismatch")
    if answer.get("task_id") != spec["task_id"] or answer.get("target_type") != spec["target_type"]:
        errors.append("Task binding mismatch")
    for row in rows:
        interval = row.get("interval", {})
        values = [interval.get(k) for k in ("level", "lo", "hi")]
        if spec["target_type"] != "classification" or "point_forecast" in row:
            values.append(row.get("point_forecast"))
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            errors.append("Missing or nonfinite numbers")
            continue
        if interval["level"] != spec["interval_level"] or interval["lo"] > interval["hi"]:
            errors.append("Invalid interval")
        if spec["target_type"] == "classification" and row.get("label") not in spec["labels"]:
            errors.append("Invalid class label")
        if not row.get("claims"):
            errors.append("Missing citations")
        for cite in row.get("claims", []):
            doc = by_doc.get(cite.get("doc_id"))
            start, end = cite.get("span_start"), cite.get("span_end")
            if doc is None or isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(doc.text):
                errors.append("Citation does not resolve to a nonempty, in-bounds span")
    if any("rank" in row for row in rows):
        ranks = [row.get("rank") for row in rows]
        if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks) or sorted(ranks) != list(range(1, len(rows) + 1)):
            errors.append("Ranks must be a complete permutation of 1..n")
    return {"structural_checks": "PASS" if not errors else "FAIL", "errors": errors,
            "future_semantics": "UNKNOWN", "production_nli": "NOT_RUN", "rankable": False}


def run(task_path, corpus_dir, out_path, diagnostics=None):
    task_path, out_path = Path(task_path), Path(out_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    spec = compile_spec(task)
    docs, excluded = load_corpus(corpus_dir, iso_date(spec["cutoff"]))
    retriever = Retriever(docs)
    rows, details, histories = [], [], []
    for entity in task["entities"]:
        row, detail, history = predict(entity, task, spec, docs, retriever.retrieve(entity, task))
        rows.append(row)
        details.append(detail)
        if history:
            histories.append({"entity_id": entity["entity_id"], **history})
    answer = {"task_id": spec["task_id"], "schema_version": task.get("schema_version", "3"),
              "target_type": spec["target_type"], "entity_predictions": rows,
              "evidence_trace": f"Verified {len(docs)} manifest-declared eligible documents; entity-grounded BM25 retrieval and transparent historical priors. Future outcome quality, interval calibration and production NLI are unverified.",
              "notes": {"production_nli": "NOT_RUN", "rankable": False, "intervals_calibrated": False}}
    checks = check_answer(answer, spec, docs)
    if checks["errors"]:
        raise ValueError("Compiled structural checks refused output: " + "; ".join(checks["errors"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answer, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    report = {"spec": spec, "entities": details, "checks": checks, "excluded_documents": excluded,
              "histories": histories, "eligible_document_count": len(docs)}
    if diagnostics:
        p = Path(diagnostics)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return answer, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", nargs="?", default="analyze", choices=["analyze"])
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    run(args.task, args.corpus, args.out, args.diagnostics)
    print(f"Wrote {args.out}; production NLI NOT RUN")


if __name__ == "__main__":
    main()
