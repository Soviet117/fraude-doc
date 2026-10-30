"""Pruebas de métricas y línea base F3 (plan §11)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw

from fraude.core.schemas import RuleFinding, Severity
from fraude.eval.baseline import run_baseline, score_findings
from fraude.eval.metrics import (
    binary_metrics,
    bootstrap_ci,
    ranking_metrics,
    select_threshold,
)
from fraude.registry.cross import RegistryEntry, save_registry

TODAY = date(2026, 10, 25)

CLEAN_FICHA = {
    "estudiante": "Quispe Mamani, José Luis",
    "dni": "12345678",
    "periodo": "2025-I",
    "programa": "Computación e Informática",
    "modulo": "Ofimática y herramientas digitales",
    "turno": "Mañana",
    "fecha_matricula": "15/03/2025",
}


def _document_image() -> Image.Image:
    image = Image.new("RGB", (1200, 1600), (235, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1200, 160), fill=(31, 56, 100))
    draw.text((80, 60), "CETPRO SAN AGUSTIN", fill="white")
    for index in range(30):
        draw.text((80, 220 + index * 35), f"Documento de prueba linea {index}", fill="black")
    return image


def _write_manifest(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "split_manifest.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _row(
    sample_id: str,
    split: str,
    *,
    authentic: bool,
    attack: str | None,
    fields: dict[str, str],
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "split": split,
        "document_type": "ficha_matricula",
        "is_authentic": authentic,
        "primary_attack": attack,
        "image_path": f"images/{sample_id}.png",
        "fields": fields,
    }


def test_binary_metrics_confusion_matrix() -> None:
    metrics = binary_metrics([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])

    assert (metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]) == (2, 1, 1, 1)
    assert metrics["precision"] == 0.6667
    assert metrics["recall"] == 0.6667
    assert metrics["f1"] == 0.6667
    assert metrics["fpr"] == 0.5
    assert metrics["accuracy"] == 0.6


def test_binary_metrics_marks_undefined_denominators() -> None:
    metrics = binary_metrics([0, 0], [0, 0])

    assert metrics["precision_defined"] is False
    assert metrics["recall_defined"] is False
    assert metrics["f1_defined"] is False
    assert metrics["fpr_defined"] is True
    assert metrics["fpr"] == 0.0


def test_ranking_metrics_perfect_separation() -> None:
    metrics = ranking_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_ranking_metrics_known_values() -> None:
    metrics = ranking_metrics([0, 1, 0, 1], [0.5, 0.4, 0.1, 0.8])

    assert metrics["roc_auc"] == 0.75
    assert metrics["pr_auc"] == 0.8333


def test_ranking_metrics_require_both_classes() -> None:
    metrics = ranking_metrics([1, 1], [0.6, 0.7])

    assert metrics == {"pr_auc": None, "roc_auc": None}


def test_bootstrap_ci_is_deterministic() -> None:
    first = bootstrap_ci([0, 0, 1, 1], [0, 0, 1, 1], "recall", n_boot=200, seed=42)
    second = bootstrap_ci([0, 0, 1, 1], [0, 0, 1, 1], "recall", n_boot=200, seed=42)

    assert first == second
    assert first["low"] is not None
    assert first["high"] is not None
    assert 0.0 <= first["low"] <= first["high"] <= 1.0


def test_bootstrap_ci_skips_undefined_metric() -> None:
    interval = bootstrap_ci([0, 0], [0, 0], "recall", n_boot=50, seed=42)

    assert interval == {"low": None, "high": None}


def test_select_threshold_prefers_f1_and_tiebreaks_near_half() -> None:
    selection = select_threshold(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.4, 0.9],
        thresholds=[0.3, 0.4, 0.5],
    )

    assert selection["threshold"] == 0.4
    assert selection["metrics"]["f1"] == 1.0


def test_score_findings_uses_severity_weights_and_caps() -> None:
    assert (
        score_findings(
            [
                RuleFinding(code="A", severity=Severity.HIGH, message=""),
                RuleFinding(code="B", severity=Severity.MEDIUM, message=""),
            ]
        )
        == 0.85
    )
    assert (
        score_findings(
            [
                RuleFinding(code="A", severity=Severity.HIGH, message=""),
                RuleFinding(code="B", severity=Severity.HIGH, message=""),
            ]
        )
        == 1.0
    )


def test_run_baseline_with_tiny_manifest(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for sample_id in ("val_auth", "val_t4", "test_auth", "test_t1", "test_t2"):
        _document_image().save(images / f"{sample_id}.png")

    missing_name = dict(CLEAN_FICHA)
    missing_name.pop("estudiante")
    changed_name = dict(CLEAN_FICHA)
    changed_name["estudiante"] = "Otra Persona"
    t2_fields = dict(CLEAN_FICHA)
    t2_fields["sello_insertado"] = "CETPRO SAN AGUSTÍN"

    manifest = _write_manifest(
        tmp_path,
        [
            _row("val_auth", "val", authentic=True, attack=None, fields=CLEAN_FICHA),
            _row("val_t4", "val", authentic=False, attack="T4", fields=missing_name),
            _row("test_auth", "test", authentic=True, attack=None, fields=CLEAN_FICHA),
            _row("test_t1", "test", authentic=False, attack="T1", fields=changed_name),
            _row("test_t2", "test", authentic=False, attack="T2", fields=t2_fields),
        ],
    )
    registry_path = tmp_path / "registry.jsonl"
    save_registry(
        [
            RegistryEntry(
                source_id="persona_0000",
                nombre="Quispe Mamani, José Luis",
                dni="12345678",
                programa="Computación e Informática",
                modulo="Ofimática y herramientas digitales",
                periodo="2025-I",
                fecha_nacimiento="15/03/1990",
            )
        ],
        registry_path,
    )
    out_path = tmp_path / "artifacts" / "f3.json"

    payload = run_baseline(
        manifest,
        registry_path,
        n_boot=100,
        seed=7,
        today=TODAY,
        out_path=out_path,
    )

    assert payload["thresholds"]["selected_on_val"] == 0.25
    assert payload["thresholds"]["oracle_on_test"] == 0.5
    fixed = payload["evaluation"]["test"]["fixed_0_5"]["metrics"]
    assert fixed["tp"] == 1
    assert fixed["fn"] == 1
    assert fixed["tn"] == 1
    assert fixed["fp"] == 0
    assert fixed["precision"] == 1.0
    assert fixed["recall"] == 0.5
    assert fixed["f1"] == 0.6667
    assert fixed["fpr"] == 0.0
    assert fixed["pr_auc"] == 0.8333
    assert fixed["roc_auc"] == 0.75
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["dataset"]["fields_source"] == "manifest_oracle"
    assert saved["limitations"]
