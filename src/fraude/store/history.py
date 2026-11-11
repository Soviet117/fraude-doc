"""Persistencia SQLite del historial de análisis (F9).

Guarda el documento, el resultado integral y la revisión humana. El payload
completo vive en JSON dentro de SQLite para que la demo pueda reconstruir el
informe sin depender de archivos temporales. El sistema nunca decide: la tabla
``feedback`` registra la revisión de una persona.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DB = Path("artifacts/history.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    filename TEXT NOT NULL,
    doctype TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    created_at TEXT NOT NULL,
    p_fake REAL,
    risk TEXT,
    calibration_method TEXT,
    model_versions TEXT,
    explanation_source TEXT,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL REFERENCES analyses(id),
    created_at TEXT NOT NULL,
    reviewer TEXT,
    label TEXT NOT NULL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_analysis ON feedback(analysis_id);
"""


class HistoryStore:
    """Acceso al historial local de análisis y revisiones."""

    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: sqlite3.Connection | None = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(self.path)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.executescript(SCHEMA)
            self._connection.commit()
        return self._connection

    def __enter__(self) -> HistoryStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def save_analysis(self, analysis: dict[str, Any]) -> int:
        """Guarda el resultado integral y devuelve el id del análisis."""
        observation = analysis.get("observation") or {}
        source = observation.get("source") or {}
        document_id = str(observation.get("document_id") or analysis.get("document_id") or "")
        if not document_id:
            raise ValueError("El análisis no incluye document_id")
        filename = str(source.get("filename") or "documento")
        doctype = str((observation.get("doctype") or {}).get("label") or "desconocido")
        fusion = analysis.get("fusor") or (observation.get("fusion") or {})
        p_fake = fusion.get("p_fake")
        risk = fusion.get("risk") or "indeterminado"
        versions = analysis.get("model_versions") or {}
        explanation = analysis.get("explanation") or {}
        now = datetime.now(UTC).isoformat()

        connection = self.connection
        connection.execute(
            """
            INSERT INTO documents (document_id, sha256, filename, doctype, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                filename = excluded.filename,
                doctype = excluded.doctype,
                last_seen = excluded.last_seen
            """,
            (
                document_id,
                document_id.removeprefix("sha256:"),
                filename,
                doctype,
                now,
                now,
            ),
        )
        cursor = connection.execute(
            """
            INSERT INTO analyses (
                document_id, created_at, p_fake, risk, calibration_method,
                model_versions, explanation_source, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                now,
                float(p_fake) if p_fake is not None else None,
                risk,
                fusion.get("calibration_method"),
                json.dumps(versions, ensure_ascii=False),
                explanation.get("source"),
                json.dumps(analysis, ensure_ascii=False),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid or 0)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT a.id, a.created_at, a.p_fake, a.risk, a.explanation_source,
                   d.filename, d.doctype
            FROM analyses a
            JOIN documents d ON d.document_id = a.document_id
            ORDER BY a.id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [dict(row) for row in rows]

    def get(self, analysis_id: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM analyses WHERE id = ?",
            (int(analysis_id),),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        payload["analysis_id"] = row["id"]
        payload["created_at"] = row["created_at"]
        return payload

    def add_feedback(
        self,
        analysis_id: int,
        *,
        label: str,
        notes: str = "",
        reviewer: str = "",
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO feedback (analysis_id, created_at, reviewer, label, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(analysis_id),
                datetime.now(UTC).isoformat(),
                reviewer,
                label,
                notes,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid or 0)

    def feedback_for(self, analysis_id: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM feedback WHERE analysis_id = ? ORDER BY id DESC",
            (int(analysis_id),),
        ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        total = self.connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        by_risk = {
            row["risk"]: row["n"]
            for row in self.connection.execute(
                "SELECT risk, COUNT(*) AS n FROM analyses GROUP BY risk"
            ).fetchall()
        }
        reviewed = self.connection.execute(
            "SELECT COUNT(DISTINCT analysis_id) FROM feedback"
        ).fetchone()[0]
        return {"total": total, "by_risk": by_risk, "reviewed": reviewed}
