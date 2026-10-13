"""SQLite persistence for cases, analyses, reviews, teaching and research data.

Design notes
------------
* One database file (``outputs/radiology_copilot.db``) holds everything.  This
  is a demo/prototype, so a single file keeps deployment to ``python app.py``.
* Large arrays never go in the database.  NIfTI volumes live on the filesystem;
  the database stores paths plus the *derived scalar* results (volumes, Dice).
* Reviewed cases are stored as **new, immutable rows**.  We never overwrite a
  model prediction in place, so the research module can always compare the
  original prediction against what the human corrected it to.
* Every write that a human is responsible for records who/what/when, because
  the whole point of the research loop is traceability.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB = PROJECT_ROOT / "outputs" / "radiology_copilot.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id               TEXT PRIMARY KEY,
    dataset_source        TEXT NOT NULL,
    modalities            TEXT NOT NULL DEFAULT '[]',
    image_paths           TEXT NOT NULL DEFAULT '{}',
    reference_mask_path   TEXT,
    prediction_mask_path  TEXT,
    corrected_mask_path   TEXT,
    model_name            TEXT,
    model_version         TEXT,
    segmentation_status   TEXT NOT NULL DEFAULT 'not_run',
    label_convention      TEXT,
    measurements          TEXT NOT NULL DEFAULT '{}',
    report_draft          TEXT,
    review_status         TEXT NOT NULL DEFAULT 'unreviewed',
    teaching_record       TEXT NOT NULL DEFAULT '{}',
    research_feedback     TEXT NOT NULL DEFAULT '{}',
    data_origin           TEXT NOT NULL DEFAULT 'unknown',
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);

-- One row per segmentation run.  Kept append-only so that model versions can
-- be compared over time.
CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id          TEXT PRIMARY KEY,
    case_id         TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    model_version   TEXT,
    device          TEXT,
    status          TEXT NOT NULL,          -- running | success | failed | skipped
    error_message   TEXT,
    duration_s      REAL,
    measurements    TEXT NOT NULL DEFAULT '{}',
    evaluation      TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases (case_id)
);

-- Append-only: an edited report creates a new revision.
CREATE TABLE IF NOT EXISTS reports (
    report_id     TEXT PRIMARY KEY,
    case_id       TEXT NOT NULL,
    revision      INTEGER NOT NULL,
    content       TEXT NOT NULL,
    status        TEXT NOT NULL,            -- draft | pending_review | approved
    generated_by  TEXT NOT NULL,            -- template | llm:<model> | human
    created_at    TEXT NOT NULL,
    approved_at   TEXT,
    FOREIGN KEY (case_id) REFERENCES cases (case_id)
);

-- Human-corrected masks.  Append-only: a correction is never edited in place,
-- so the original model prediction and every human revision stay comparable.
-- `source` separates a genuine user upload from a demo replay of the dataset
-- reference annotation; the two must never be aggregated together.
CREATE TABLE IF NOT EXISTS corrections (
    correction_id  TEXT PRIMARY KEY,
    case_id        TEXT NOT NULL,
    revision       INTEGER NOT NULL,
    mask_path      TEXT NOT NULL,
    source         TEXT NOT NULL,          -- user_upload | demo_reference_simulation
    validation     TEXT NOT NULL DEFAULT '{}',
    metrics        TEXT NOT NULL DEFAULT '{}',
    notes          TEXT,
    created_at     TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases (case_id)
);

CREATE TABLE IF NOT EXISTS teaching_sessions (
    session_id      TEXT PRIMARY KEY,
    case_id         TEXT NOT NULL,
    answer_text     TEXT,
    submitted_mask  TEXT,
    evaluation      TEXT NOT NULL DEFAULT '{}',
    feedback        TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases (case_id)
);

CREATE TABLE IF NOT EXISTS research_feedback (
    feedback_id    TEXT PRIMARY KEY,
    case_id        TEXT NOT NULL,
    dataset_source TEXT,
    model_name     TEXT,
    model_version  TEXT,
    error_type     TEXT NOT NULL,
    notes          TEXT,
    metrics        TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases (case_id)
);

-- Audit trail for the agent.  Stores the tools that actually ran, so we can
-- prove tool execution rather than trusting the model's narration.
CREATE TABLE IF NOT EXISTS agent_traces (
    trace_id      TEXT PRIMARY KEY,
    case_id       TEXT,
    user_message  TEXT NOT NULL,
    mode          TEXT NOT NULL,            -- llm_tool_calling | fixed_workflow
    tool_calls    TEXT NOT NULL DEFAULT '[]',
    final_answer  TEXT,
    model         TEXT,
    created_at    TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Store:
    """Small synchronous wrapper around SQLite.

    Gradio calls the callbacks from a thread pool, so a lock serialises writes
    and ``check_same_thread=False`` lets a single connection be shared.
    """

    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- plumbing --------------------------------------------------------
    @contextmanager
    def _cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            finally:
                cur.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- cases -----------------------------------------------------------
    def upsert_case(self, case_id: str, **fields: Any) -> None:
        """Insert or update a case, JSON-encoding list/dict fields."""
        payload = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                   for k, v in fields.items()}
        with self._cursor() as cur:
            row = cur.execute("SELECT case_id FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None:
                payload.setdefault("dataset_source", "unknown")
                payload["case_id"] = case_id
                payload["created_at"] = _now()
                payload["updated_at"] = _now()
                cols = ", ".join(payload)
                marks = ", ".join("?" for _ in payload)
                cur.execute(f"INSERT INTO cases ({cols}) VALUES ({marks})", tuple(payload.values()))
            else:
                payload["updated_at"] = _now()
                sets = ", ".join(f"{k} = ?" for k in payload)
                cur.execute(f"UPDATE cases SET {sets} WHERE case_id = ?",
                            (*payload.values(), case_id))

    def get_case(self, case_id: str) -> dict | None:
        with self._cursor() as cur:
            row = cur.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        return _decode_case(row) if row else None

    def list_cases(self) -> list[dict]:
        with self._cursor() as cur:
            rows = cur.execute("SELECT * FROM cases ORDER BY case_id").fetchall()
        return [_decode_case(r) for r in rows]

    # -- analysis runs ---------------------------------------------------
    def record_analysis_run(self, case_id: str, model_name: str, status: str, **fields: Any) -> str:
        run_id = _new_id("run")
        payload = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                   for k, v in fields.items()}
        payload.update(run_id=run_id, case_id=case_id, model_name=model_name,
                       status=status, created_at=_now())
        cols = ", ".join(payload)
        marks = ", ".join("?" for _ in payload)
        with self._cursor() as cur:
            cur.execute(f"INSERT INTO analysis_runs ({cols}) VALUES ({marks})", tuple(payload.values()))
        return run_id

    def list_analysis_runs(self, case_id: str | None = None, limit: int = 50) -> list[dict]:
        # Ordered by rowid as well as created_at: timestamps have one-second
        # resolution, so two runs started in the same second would otherwise
        # come back in arbitrary order and "the latest run" would be a coin
        # flip.  rowid is monotonic in insertion order.
        q = "SELECT * FROM analysis_runs"
        args: tuple = ()
        if case_id:
            q += " WHERE case_id = ?"
            args = (case_id,)
        q += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        with self._cursor() as cur:
            rows = cur.execute(q, (*args, limit)).fetchall()
        return [_jsonify(r) for r in rows]

    # -- reports ---------------------------------------------------------
    def save_report(self, case_id: str, content: str, status: str, generated_by: str,
                    approve: bool = False) -> str:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT COALESCE(MAX(revision), 0) AS r FROM reports WHERE case_id = ?", (case_id,)
            ).fetchone()
            revision = int(row["r"]) + 1
            report_id = _new_id("rep")
            cur.execute(
                "INSERT INTO reports (report_id, case_id, revision, content, status, "
                "generated_by, created_at, approved_at) VALUES (?,?,?,?,?,?,?,?)",
                (report_id, case_id, revision, content, status, generated_by,
                 _now(), _now() if approve else None),
            )
            review_status = "approved" if approve else ("pending_review" if status == "pending_review" else "draft")
            cur.execute(
                "UPDATE cases SET report_draft = ?, review_status = ?, updated_at = ? WHERE case_id = ?",
                (content, review_status, _now(), case_id),
            )
        return report_id

    def latest_report(self, case_id: str) -> dict | None:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT * FROM reports WHERE case_id = ? ORDER BY revision DESC LIMIT 1", (case_id,)
            ).fetchone()
        return dict(row) if row else None

    def report_history(self, case_id: str) -> list[dict]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM reports WHERE case_id = ? ORDER BY revision DESC", (case_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- teaching --------------------------------------------------------
    def record_teaching_session(self, case_id: str, answer_text: str | None,
                                submitted_mask: str | None, evaluation: dict,
                                feedback: str | None) -> str:
        sid = _new_id("teach")
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO teaching_sessions (session_id, case_id, answer_text, submitted_mask,"
                " evaluation, feedback, created_at) VALUES (?,?,?,?,?,?,?)",
                (sid, case_id, answer_text, submitted_mask,
                 json.dumps(evaluation, ensure_ascii=False), feedback, _now()),
            )
            cur.execute("UPDATE cases SET teaching_record = ?, updated_at = ? WHERE case_id = ?",
                        (json.dumps(evaluation, ensure_ascii=False), _now(), case_id))
        return sid

    def list_teaching_sessions(self, case_id: str | None = None, limit: int = 100) -> list[dict]:
        q = "SELECT * FROM teaching_sessions"
        args: tuple = ()
        if case_id:
            q += " WHERE case_id = ?"
            args = (case_id,)
        q += " ORDER BY created_at DESC LIMIT ?"
        with self._cursor() as cur:
            rows = cur.execute(q, (*args, limit)).fetchall()
        return [_jsonify(r) for r in rows]

    # -- human corrections ----------------------------------------------
    def record_correction(self, case_id: str, revision: int, mask_path: str,
                          source: str, validation: dict | None = None,
                          metrics_payload: dict | None = None,
                          notes: str = "") -> str:
        cid = _new_id("corr")
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO corrections (correction_id, case_id, revision, mask_path,"
                " source, validation, metrics, notes, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (cid, case_id, revision, mask_path, source,
                 json.dumps(validation or {}, ensure_ascii=False, default=str),
                 json.dumps(metrics_payload or {}, ensure_ascii=False, default=str),
                 notes, _now()),
            )
        return cid

    def list_corrections(self, case_id: str | None = None, limit: int = 100) -> list[dict]:
        q = "SELECT * FROM corrections"
        args: tuple = ()
        if case_id:
            q += " WHERE case_id = ?"
            args = (case_id,)
        q += " ORDER BY created_at DESC, revision DESC, rowid DESC LIMIT ?"
        with self._cursor() as cur:
            rows = cur.execute(q, (*args, limit)).fetchall()
        return [_jsonify(r) for r in rows]

    def next_correction_revision(self, case_id: str) -> int:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT COALESCE(MAX(revision), 0) AS r FROM corrections WHERE case_id = ?",
                (case_id,)).fetchone()
        return int(row["r"]) + 1

    # -- research feedback ----------------------------------------------
    def record_feedback(self, case_id: str, error_type: str, **fields: Any) -> str:
        fid = _new_id("fb")
        payload = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                   for k, v in fields.items()}
        payload.setdefault("dataset_source", None)
        payload.setdefault("model_name", None)
        payload.setdefault("model_version", None)
        payload.setdefault("notes", None)
        payload.setdefault("metrics", "{}")
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO research_feedback (feedback_id, case_id, dataset_source, model_name,"
                " model_version, error_type, notes, metrics, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (fid, case_id, payload["dataset_source"], payload["model_name"],
                 payload["model_version"], error_type, payload["notes"],
                 payload["metrics"], _now()),
            )
        return fid

    def list_feedback(self, limit: int = 500) -> list[dict]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM research_feedback ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_jsonify(r) for r in rows]

    def feedback_counts(self) -> dict[str, int]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT error_type, COUNT(*) AS n FROM research_feedback GROUP BY error_type"
            ).fetchall()
        return {r["error_type"]: int(r["n"]) for r in rows}

    # -- agent traces ----------------------------------------------------
    def record_agent_trace(self, case_id: str | None, user_message: str, mode: str,
                           tool_calls: list[dict], final_answer: str | None,
                           model: str | None = None) -> str:
        tid = _new_id("trace")
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO agent_traces (trace_id, case_id, user_message, mode, tool_calls,"
                " final_answer, model, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (tid, case_id, user_message, mode,
                 json.dumps(tool_calls, ensure_ascii=False, default=str),
                 final_answer, model, _now()),
            )
        return tid

    def list_agent_traces(self, limit: int = 50) -> list[dict]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM agent_traces ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_jsonify(r) for r in rows]

    # -- aggregates used by the dashboards -------------------------------
    def counts(self) -> dict[str, int]:
        with self._cursor() as cur:
            out = {}
            for key, sql in {
                "cases": "SELECT COUNT(*) n FROM cases",
                "analyses": "SELECT COUNT(*) n FROM analysis_runs WHERE status = 'success'",
                "feedback": "SELECT COUNT(*) n FROM research_feedback",
                "teaching": "SELECT COUNT(*) n FROM teaching_sessions",
                "approved": "SELECT COUNT(*) n FROM reports WHERE status = 'approved'",
            }.items():
                out[key] = int(cur.execute(sql).fetchone()["n"])
        return out

    def evaluated_cases(self) -> list[dict]:
        """Successful runs carrying a real Dice evaluation, newest first.

        The ordering is explicit and total (``created_at`` then ``rowid``) so
        callers can rely on the first row for a case being its latest run.
        """
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM analysis_runs WHERE status = 'success' AND evaluation != '{}'"
                " ORDER BY created_at DESC, rowid DESC"
            ).fetchall()
        return [_jsonify(r) for r in rows]


def _decode_case(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("modalities", "image_paths", "measurements", "teaching_record", "research_feedback"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except json.JSONDecodeError:
                d[key] = {} if key != "modalities" else []
    return d


def _jsonify(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, str) and v[:1] in "[{":
            try:
                d[k] = json.loads(v)
            except json.JSONDecodeError:
                pass
    return d


def seed_cases_from_registry(store: Store, cases: Iterable[Any]) -> int:
    """Make sure every discovered case exists as a row. Returns rows touched."""
    n = 0
    for c in cases:
        if store.get_case(c.case_id) is None:
            store.upsert_case(
                c.case_id,
                dataset_source=c.dataset_source,
                modalities=list(c.modalities),
                image_paths=c.image_paths,
                reference_mask_path=c.reference_mask_path,
                label_convention=c.label_convention,
                data_origin=c.data_origin,
                segmentation_status="not_run",
                review_status="unreviewed",
            )
        n += 1
    return n
