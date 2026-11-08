"""Generador de manipulaciones sintéticas T1-T8 (F2.3).

Cada ataque escribe una imagen derivada, una máscara cuando la manipulación es
local y un registro con el campo original y el valor alterado. Los ataques
semánticos (T7, T8) no modifican píxeles: se etiquetan con ``semantic_flags`` y
con la caja del campo afectado, sin máscara, para no entrenar localización con
señales falsas. T6 (documento fabricado) usa una máscara de página completa.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageStat

from fraude.ingest.hashing import sha256_file
from fraude.synth.data import Identity
from fraude.synth.dataset import ATTACK_TYPES, DocumentRecord, Region
from fraude.synth.render import render_document

NAVY = (31, 56, 100)
RED = (140, 30, 30)
GRAY = (90, 90, 90)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

_TEXT_EXCLUDED = {"foto", "sello", "firma", "firma_docente", "firma_director"}
_STAMP_KEYS = ("sello", "firma", "firma_docente", "firma_director")

_T1_PREFERENCES: dict[str, tuple[str, ...]] = {
    "dni": ("numero", "nombres", "fecha_nacimiento"),
    "certificado_estudios": ("estudiante", "promedio", "fecha_emision"),
    "partida_nacimiento": ("nombre", "fecha_nacimiento", "numero_acta"),
    "ficha_matricula": ("estudiante", "programa", "modulo"),
    "acta_evaluacion": ("nota_0", "estudiante_0", "nota_1"),
    "certificado_modulo": ("estudiante", "nota", "modulo"),
    "titulo_auxiliar_tecnico": ("estudiante", "programa", "resolucion"),
}

_T8_PREFERENCES: dict[str, tuple[str, ...]] = {
    "dni": ("fecha_emision", "fecha_vencimiento"),
    "certificado_estudios": ("promedio", "anio_egreso"),
    "partida_nacimiento": ("fecha_registro", "numero_acta"),
    "ficha_matricula": ("periodo", "modulo"),
    "acta_evaluacion": ("nota_0", "nota_1"),
    "certificado_modulo": ("nota", "horas"),
    "titulo_auxiliar_tecnico": ("resolucion",),
}


def generate_attacks(
    authentic: list[DocumentRecord],
    out_dir: Path,
    *,
    source_dir: Path | None = None,
    seed: int = 42,
    copies: int = 1,
    dpi: int = 150,
    attacks: Sequence[str] | None = None,
    per_attack: Mapping[str, int] | None = None,
) -> list[DocumentRecord]:
    """Genera muestras manipuladas T1-T8 cuando hay un padre apto.

    ``copies`` fija cuántas copias se generan por ataque. ``attacks`` permite
    limitar el subconjunto de familias (útil para lotes de humo) y
    ``per_attack`` sobrescribe el número de copias de una familia concreta.
    """
    if not authentic:
        raise ValueError("Se necesita al menos un documento auténtico")
    if copies < 1:
        raise ValueError("copies debe ser mayor que cero")

    selected = tuple(attacks) if attacks is not None else ATTACK_TYPES
    unknown = sorted(set(selected) - set(ATTACK_TYPES))
    if unknown:
        raise ValueError(f"Ataques no soportados: {', '.join(unknown)}")
    if len(set(selected)) != len(selected):
        raise ValueError("attacks no puede contener duplicados")

    counts: dict[str, int] = {}
    for attack_type in selected:
        value = per_attack.get(attack_type, copies) if per_attack is not None else copies
        if value < 0:
            raise ValueError(f"El número de copias de {attack_type} no puede ser negativo")
        counts[attack_type] = value
    if per_attack is not None:
        extra = sorted(set(per_attack) - set(selected))
        if extra:
            raise ValueError(f"per_attack incluye ataques no seleccionados: {', '.join(extra)}")

    source_dir = Path(source_dir) if source_dir is not None else Path(out_dir)
    out_dir = Path(out_dir)
    rng = random.Random(seed)
    records: list[DocumentRecord] = []

    for attack_type in selected:
        parents = _candidates(authentic, attack_type)
        attack_index = ATTACK_TYPES.index(attack_type)
        for copy_index in range(counts[attack_type]):
            if not parents:
                break
            parent = parents[(copy_index + attack_index) % len(parents)]
            donor = _donor_for(parent, authentic) if attack_type == "T7" else None
            if attack_type == "T7" and donor is None:
                break
            records.append(
                apply_attack(
                    parent,
                    attack_type,
                    out_dir,
                    source_dir=source_dir,
                    rng=rng,
                    donor=donor,
                    copy_index=copy_index,
                    dpi=dpi,
                )
            )

    write_attacks_manifest(records, out_dir, seed=seed, copies=copies, per_attack=counts)
    return records


def apply_attack(
    parent: DocumentRecord,
    attack_type: str,
    out_dir: Path,
    *,
    source_dir: Path | None = None,
    rng: random.Random | None = None,
    donor: DocumentRecord | None = None,
    copy_index: int = 0,
    dpi: int = 150,
) -> DocumentRecord:
    """Aplica un ataque a un documento auténtico y devuelve su registro."""
    if attack_type not in ATTACK_TYPES:
        raise ValueError(f"Ataque no soportado: {attack_type}")
    rng = rng or random.Random(parent.seed)
    source_dir = Path(source_dir) if source_dir is not None else Path(out_dir)
    out_dir = Path(out_dir)
    sample_id = f"{parent.sample_id}_{attack_type.lower()}"
    if copy_index:
        sample_id += f"_c{copy_index}"

    if attack_type == "T7":
        if donor is None:
            raise ValueError("T7 requiere un documento donante")
        return _reuse_document(parent, donor, out_dir, source_dir, sample_id)

    if attack_type == "T6":
        return _forge_document(parent, out_dir, sample_id)

    if attack_type == "T8":
        return _semantic_inconsistency(parent, out_dir, sample_id, dpi=dpi)

    image = Image.open(source_dir / parent.image_path).convert("RGB")
    if attack_type == "T1":
        image, mask, region, fields = _substitute_text(image, parent, rng)
    elif attack_type == "T2":
        image, mask, region, fields = _insert_stamp(image, parent, rng)
    elif attack_type == "T3":
        image, mask, region, fields = _copy_move(image, parent)
    elif attack_type == "T4":
        image, mask, region, fields = _erase_field(image, parent, rng)
    elif attack_type == "T5":
        image, mask, region, fields = _replace_photo(image, parent, rng)
    else:  # pragma: no cover - ATTACK_TYPES lo hace inalcanzable
        raise ValueError(f"Ataque no soportado: {attack_type}")

    return _save_tampered(
        parent,
        out_dir,
        sample_id,
        attack_type,
        image=image,
        mask=mask,
        region=region,
        fields=fields,
    )


def _substitute_text(
    image: Image.Image,
    parent: DocumentRecord,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, Region, dict[str, Any]]:
    field = _pick_field(
        parent, _T1_PREFERENCES.get(parent.document_type, ()), exclude=_TEXT_EXCLUDED
    )
    if field is None:
        raise ValueError(f"{parent.sample_id} no tiene campos de texto para T1")
    bbox = _expand_bbox(parent.field_boxes[field], image.size, pad=2)
    original = str(parent.fields.get(field, ""))
    altered = _fake_value(field, original, rng)

    color = _background_color(image, bbox)
    draw = ImageDraw.Draw(image)
    draw.rectangle(_xyxy(bbox), fill=color)
    _draw_text_in_box(draw, bbox, altered, color=BLACK)

    mask = _new_mask(image.size)
    ImageDraw.Draw(mask).rectangle(_xyxy(bbox), fill=255)
    fields = dict(parent.fields)
    fields[field] = altered
    region = Region(
        attack_type="T1",
        bbox=bbox,
        field=field,
        original_value=original,
        altered_value=altered,
    )
    return image, mask, region, fields


def _insert_stamp(
    image: Image.Image,
    parent: DocumentRecord,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, Region, dict[str, Any]]:
    anchor = _stamp_anchor(parent, image.size)
    radius = 34
    x = min(max(anchor[0], radius + 8), image.width - radius - 8)
    y = min(max(anchor[1], radius + 8), image.height - radius - 8)
    _draw_stamp(image, x, y, radius, rng)

    bbox = (x - radius - 3, y - radius - 3, 2 * radius + 6, 2 * radius + 6)
    mask = _new_mask(image.size)
    ImageDraw.Draw(mask).ellipse(_xyxy(bbox), fill=255)
    fields = dict(parent.fields)
    fields["sello_insertado"] = "CETPRO SAN AGUSTÍN"
    region = Region(
        attack_type="T2",
        bbox=bbox,
        field="sello_insertado",
        original_value=None,
        altered_value="sello",
    )
    return image, mask, region, fields


def _copy_move(
    image: Image.Image,
    parent: DocumentRecord,
) -> tuple[Image.Image, Image.Image, Region, dict[str, Any]]:
    source_key = next((key for key in (*_STAMP_KEYS, "foto") if key in parent.field_boxes), None)
    if source_key is None:
        source_key = _pick_field(parent, (), exclude=_TEXT_EXCLUDED)
    if source_key is None:
        raise ValueError(f"{parent.sample_id} no tiene región para copy-move")

    source = _expand_bbox(parent.field_boxes[source_key], image.size, pad=2)
    sx, sy, sw, sh = source
    crop = image.crop(_xyxy(source))

    if source_key == "foto":
        dx = image.width - sw - 45
        dy = image.height - sh - 45
    else:
        dx = sx + sw + 40
        if dx + sw > image.width - 20:
            dx = sx - sw - 40
        dy = sy - 45
        if dy < 20:
            dy = sy + sh + 35
    dx = min(max(dx, 15), image.width - sw - 15)
    dy = min(max(dy, 15), image.height - sh - 15)
    image.paste(crop, (dx, dy))

    bbox = (dx, dy, sw, sh)
    mask = _new_mask(image.size)
    ImageDraw.Draw(mask).rectangle(_xyxy(bbox), fill=255)
    fields = dict(parent.fields)
    fields[f"{source_key}_copiado"] = "duplicado"
    region = Region(
        attack_type="T3",
        bbox=bbox,
        field=f"{source_key}_copiado",
        original_value=None,
        altered_value="duplicado",
    )
    return image, mask, region, fields


def _erase_field(
    image: Image.Image,
    parent: DocumentRecord,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, Region, dict[str, Any]]:
    field = _pick_field(
        parent, _T1_PREFERENCES.get(parent.document_type, ()), exclude=_TEXT_EXCLUDED
    )
    if field is None:
        raise ValueError(f"{parent.sample_id} no tiene campos para T4")
    bbox = _expand_bbox(parent.field_boxes[field], image.size, pad=3)
    original = str(parent.fields.get(field, ""))

    color = _background_color(image, bbox)
    draw = ImageDraw.Draw(image)
    draw.rectangle(_xyxy(bbox), fill=color)
    # Ruido leve para simular un relleno imperfecto, no un blanco perfecto.
    noise = rng.randint(0, 4)
    if noise:
        for _ in range(noise):
            px = rng.randint(bbox[0], bbox[0] + bbox[2] - 1)
            py = rng.randint(bbox[1], bbox[1] + bbox[3] - 1)
            draw.point(
                (px, py),
                fill=(max(0, color[0] - 6), max(0, color[1] - 6), max(0, color[2] - 6)),
            )

    mask = _new_mask(image.size)
    ImageDraw.Draw(mask).rectangle(_xyxy(bbox), fill=255)
    fields = dict(parent.fields)
    fields.pop(field, None)
    region = Region(
        attack_type="T4",
        bbox=bbox,
        field=field,
        original_value=original,
        altered_value=None,
    )
    return image, mask, region, fields


def _replace_photo(
    image: Image.Image,
    parent: DocumentRecord,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, Region, dict[str, Any]]:
    if "foto" not in parent.field_boxes:
        raise ValueError(f"{parent.sample_id} no tiene fotografía para T5")
    bbox = parent.field_boxes["foto"]
    _draw_face(image, bbox, rng)

    mask = _new_mask(image.size)
    ImageDraw.Draw(mask).rectangle(_xyxy(bbox), fill=255)
    fields = dict(parent.fields)
    fields["foto"] = "reemplazada"
    region = Region(
        attack_type="T5",
        bbox=bbox,
        field="foto",
        original_value="foto_original",
        altered_value="foto_reemplazada",
    )
    return image, mask, region, fields


def _forge_document(
    parent: DocumentRecord,
    out_dir: Path,
    sample_id: str,
) -> DocumentRecord:
    image = Image.new("RGB", parent.page_size, WHITE)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle((0, 0, width, int(height * 0.12)), fill=RED)
    _draw_centered(
        draw, width / 2, int(height * 0.045), "INSTITUTO TÉCNICO PARTICULAR", 28, WHITE, bold=True
    )
    _draw_centered(
        draw, width / 2, int(height * 0.08), "SAN MARTÍN — SEDE NUEVA CAJAMARCA", 18, WHITE
    )

    title = parent.document_type.replace("_", " ").upper()
    _draw_centered(draw, width / 2, int(height * 0.22), title, 26, RED, bold=True)
    draw.line((40, int(height * 0.25), width - 40, int(height * 0.25)), fill=RED, width=3)

    y = int(height * 0.32)
    for key, value in parent.fields.items():
        if key in {"foto", "sello", "firma", "firma_docente", "firma_director"}:
            continue
        _draw_text(draw, (50, y), f"{key}: {value}", 18, BLACK)
        y += 34
        if y > height - 160:
            break

    _draw_centered(
        draw,
        width / 2,
        height - 100,
        "Documento emitido fuera del registro institucional",
        16,
        GRAY,
    )
    draw.ellipse(
        (width - 190, height - 190, width - 70, height - 70),
        outline=RED,
        width=4,
    )
    _draw_centered(draw, width - 130, height - 140, "SELLO", 14, RED, bold=True)
    _draw_centered(draw, width - 130, height - 120, "PARTICULAR", 12, RED)

    mask = Image.new("L", image.size, 255)
    region = Region(
        attack_type="T6",
        bbox=(0, 0, width, height),
        field=None,
        original_value=parent.document_type,
        altered_value="documento_fabricado",
    )
    return _save_tampered(
        parent,
        out_dir,
        sample_id,
        "T6",
        image=image,
        mask=mask,
        region=region,
        fields=dict(parent.fields),
        field_boxes={},
        template_version="forged",
        semantic_flags=("full_fabrication",),
    )


def _reuse_document(
    parent: DocumentRecord,
    donor: DocumentRecord,
    out_dir: Path,
    source_dir: Path,
    sample_id: str,
) -> DocumentRecord:
    image_rel = f"images/{sample_id}.png"
    image_path = out_dir / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_dir / donor.image_path, image_path)

    identity_key = next(
        (
            key
            for key in ("estudiante", "nombre", "nombres", "numero", "dni")
            if key in donor.field_boxes
        ),
        None,
    )
    regions: tuple[Region, ...] = ()
    if identity_key is not None:
        regions = (
            Region(
                attack_type="T7",
                bbox=donor.field_boxes[identity_key],
                field=identity_key,
                original_value=str(donor.fields.get(identity_key, "")),
                altered_value=str(parent.fields.get(identity_key, "")),
            ),
        )

    return DocumentRecord(
        sample_id=sample_id,
        document_id=sha256_file(image_path),
        source_id=donor.source_id,
        document_type=donor.document_type,
        template_version=donor.template_version,
        variant="tampered",
        parent_id=donor.document_id,
        is_authentic=False,
        attack_types=("T7",),
        primary_attack="T7",
        pdf_path=None,
        image_path=image_rel,
        mask_paths=(),
        fields=dict(donor.fields),
        field_boxes=donor.field_boxes,
        page_size=donor.page_size,
        regions=regions,
        semantic_flags=("document_reuse",),
        capture_type=donor.capture_type,
        seed=parent.seed,
    )


def _semantic_inconsistency(
    parent: DocumentRecord,
    out_dir: Path,
    sample_id: str,
    *,
    dpi: int,
) -> DocumentRecord:
    field = _pick_field(parent, _T8_PREFERENCES.get(parent.document_type, ()))
    if field is None:
        field = next(iter(parent.field_boxes), None)
    if field is None:
        raise ValueError(f"{parent.sample_id} no tiene campos para T8")

    original = str(parent.fields.get(field, ""))
    altered = _semantic_value(field, original)
    altered_fields = dict(parent.fields)
    altered_fields[field] = altered

    base = render_document(
        sample_id=sample_id,
        document_type=parent.document_type,
        identity=_identity_from_record(parent),
        fields=altered_fields,
        out_dir=out_dir,
        seed=parent.seed,
        dpi=dpi,
    )
    bbox = base.field_boxes.get(field, (0, 0, 1, 1))
    return replace(
        base,
        variant="tampered",
        parent_id=parent.document_id,
        is_authentic=False,
        attack_types=("T8",),
        primary_attack="T8",
        mask_paths=(),
        regions=(
            Region(
                attack_type="T8",
                bbox=bbox,
                field=field,
                original_value=original,
                altered_value=altered,
            ),
        ),
        semantic_flags=("semantic_inconsistency", f"invalid_{field}"),
    )


def _save_tampered(
    parent: DocumentRecord,
    out_dir: Path,
    sample_id: str,
    attack_type: str,
    *,
    image: Image.Image,
    mask: Image.Image | None,
    region: Region | None,
    fields: dict[str, Any],
    field_boxes: dict[str, tuple[int, int, int, int]] | None = None,
    template_version: str | None = None,
    semantic_flags: tuple[str, ...] = (),
) -> DocumentRecord:
    out_dir = Path(out_dir)
    image_rel = f"images/{sample_id}.png"
    image_path = out_dir / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    dpi_info = image.info.get("dpi", (150, 150))
    image.save(image_path, format="PNG", dpi=dpi_info)

    mask_paths: tuple[str, ...] = ()
    if mask is not None:
        mask_rel = f"masks/{sample_id}.png"
        mask_path = out_dir / mask_rel
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_path, format="PNG")
        mask_paths = (mask_rel,)
        if region is not None and region.mask_path is None:
            region = replace(region, mask_path=mask_rel)

    return DocumentRecord(
        sample_id=sample_id,
        document_id=sha256_file(image_path),
        source_id=parent.source_id,
        document_type=parent.document_type,
        template_version=template_version or parent.template_version,
        variant="tampered",
        parent_id=parent.document_id,
        is_authentic=False,
        attack_types=(attack_type,),
        primary_attack=attack_type,
        pdf_path=None,
        image_path=image_rel,
        mask_paths=mask_paths,
        fields=fields,
        field_boxes=parent.field_boxes if field_boxes is None else field_boxes,
        page_size=image.size,
        regions=(region,) if region is not None else (),
        semantic_flags=semantic_flags,
        capture_type=parent.capture_type,
        seed=parent.seed,
    )


def _candidates(records: list[DocumentRecord], attack_type: str) -> list[DocumentRecord]:
    if attack_type == "T5":
        return [
            record
            for record in records
            if record.document_type == "dni" and "foto" in record.field_boxes
        ]
    if attack_type in {"T2", "T3"}:
        return [
            record
            for record in records
            if any(key in record.field_boxes for key in (*_STAMP_KEYS, "foto"))
        ]
    if attack_type == "T7":
        return [record for record in records if _donor_for(record, records) is not None]
    return list(records)


def _donor_for(parent: DocumentRecord, records: list[DocumentRecord]) -> DocumentRecord | None:
    same_type = [
        record
        for record in records
        if record.source_id != parent.source_id and record.document_type == parent.document_type
    ]
    if same_type:
        return same_type[0]
    different_person = [record for record in records if record.source_id != parent.source_id]
    return different_person[0] if different_person else None


def _pick_field(
    parent: DocumentRecord,
    preferred: tuple[str, ...],
    *,
    exclude: set[str] | None = None,
) -> str | None:
    excluded = exclude or set()
    for key in preferred:
        if key in parent.field_boxes and key not in excluded:
            return key
    for key in parent.field_boxes:
        if key not in excluded:
            return key
    return None


def _fake_value(field: str, original: str, rng: random.Random) -> str:
    if field.startswith("fecha"):
        return "01/01/1999"
    if field == "sexo":
        return "F" if original == "M" else "M"
    if field == "estado_civil":
        return "Casado(a)"
    if field in {"numero", "dni"} or field.startswith("dni_"):
        return f"{rng.randint(10_000_000, 99_999_999)}"
    if field == "numero_acta":
        return f"{rng.randint(100_000, 999_999)}"
    if field == "resolucion":
        return "RD-9999-2025-CETPRO-SA"
    if field == "promedio":
        return "20.0"
    if field in {"nota", "horas", "anio_egreso"} or field.startswith("nota_"):
        return "20" if field != "horas" else "999"
    if field == "turno":
        return "Noche"
    if field == "periodo":
        return "2026-I"
    if field == "sede":
        return "Sede alterada"
    if field == "nombres":
        return "Carlos Alberto"
    if field == "apellidos":
        return "Mendoza Ríos"
    if field in {"estudiante", "nombre"} or field.startswith("estudiante_"):
        return "Mendoza Ríos, Carlos Alberto"
    if field == "institucion":
        return "I.E. Alterada"
    if field == "direccion":
        return "Jr. Falsa 123"
    if field == "programa":
        return "Contabilidad"
    if field == "modulo":
        return "Módulo alterado"
    if field == "docente":
        return "Prof. Alterado"
    if field == "director":
        return "Director Alterado"
    if field in {"padre", "madre"}:
        return "Familiar Alterado"
    return "ALTERADO"


def _semantic_value(field: str, original: str) -> str:
    if field.startswith("fecha"):
        return "01/01/1900"
    if field == "promedio":
        return "25.0"
    if field == "nota" or field.startswith("nota_"):
        return "25"
    if field == "horas":
        return "-40"
    if field == "anio_egreso":
        return "1800"
    if field == "periodo":
        return "1999-I"
    if field == "numero_acta":
        return "000000"
    if field == "resolucion":
        return "SIN RESOLUCIÓN"
    if field == "modulo":
        return "Módulo inexistente"
    return f"{original} (inválido)"


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    page_size: tuple[int, int],
    *,
    pad: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    page_width, page_height = page_size
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(page_width, x + width + pad)
    bottom = min(page_height, y + height + pad)
    return left, top, max(1, right - left), max(1, bottom - top)


def _xyxy(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Convierte ``(x, y, ancho, alto)`` a ``(x1, y1, x2, y2)`` para PIL."""
    x, y, width, height = bbox
    return x, y, x + width, y + height


def _background_color(image: Image.Image, bbox: tuple[int, int, int, int]) -> tuple[int, int, int]:
    x, y, width, height = bbox
    border = max(1, min(width, height) // 4)
    regions = [
        image.crop((x, y, x + width, y + border)),
        image.crop((x, y + height - border, x + width, y + height)),
        image.crop((x, y, x + border, y + height)),
        image.crop((x + width - border, y, x + width, y + height)),
    ]
    pixels: list[tuple[int, int, int]] = []
    for region in regions:
        median = ImageStat.Stat(region).median
        pixels.append((int(median[0]), int(median[1]), int(median[2])))
    if not pixels:
        return WHITE
    channels = list(zip(*pixels, strict=True))
    return tuple(sorted(channel)[len(channel) // 2] for channel in channels)


def _draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    text: str,
    *,
    color: tuple[int, int, int],
    bold: bool = False,
) -> None:
    x, y, width, height = bbox
    font, text_bbox = _fit_font(text, width - 2, height - 2, bold=bold)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    draw.text(
        (
            x + (width - text_width) / 2 - text_bbox[0],
            y + (height - text_height) / 2 - text_bbox[1],
        ),
        text,
        font=font,
        fill=color,
    )


def _fit_font(
    text: str,
    max_width: int,
    max_height: int,
    *,
    bold: bool = False,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, tuple[int, int, int, int]]:
    size = max(8, int(max_height * 1.1))
    while size >= 6:
        font = ImageFont.truetype(str(_font_path(bold=bold)), size)
        bbox = font.getbbox(text)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            return font, bbox
        size -= 1
    font = ImageFont.truetype(str(_font_path(bold=bold)), 6)
    return font, font.getbbox(text)


def _font_path(*, bold: bool = False) -> Path:
    import reportlab

    return Path(reportlab.__file__).parent / "fonts" / ("VeraBd.ttf" if bold else "Vera.ttf")


def _draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    size: int,
    color: tuple[int, int, int],
    *,
    bold: bool = False,
) -> None:
    font = ImageFont.truetype(str(_font_path(bold=bold)), size)
    draw.text(position, text, font=font, fill=color)


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    text: str,
    size: int,
    color: tuple[int, int, int],
    *,
    bold: bool = False,
) -> None:
    font = ImageFont.truetype(str(_font_path(bold=bold)), size)
    bbox = font.getbbox(text)
    draw.text((center_x - (bbox[2] - bbox[0]) / 2, y), text, font=font, fill=color)


def _draw_stamp(image: Image.Image, x: int, y: int, radius: int, rng: random.Random) -> None:
    draw = ImageDraw.Draw(image)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=NAVY, width=3)
    draw.ellipse(
        (x - radius + 5, y - radius + 5, x + radius - 5, y + radius - 5), outline=NAVY, width=1
    )
    _draw_centered(draw, x, y - 18, "CETPRO", 10, NAVY, bold=True)
    _draw_centered(draw, x, y - 2, "SAN AGUSTÍN", 8, NAVY, bold=True)
    _draw_centered(draw, x, y + 12, "REGISTRO", 6, NAVY)
    # Pequeña irregularidad del trazo para no dejarlo perfectamente geométrico.
    offset = rng.randint(-2, 2)
    draw.arc(
        (x - radius + 2 + offset, y - radius + 2, x + radius - 2 + offset, y + radius - 2),
        20,
        320,
        fill=NAVY,
        width=1,
    )


def _draw_face(image: Image.Image, bbox: tuple[int, int, int, int], rng: random.Random) -> None:
    x, y, width, height = bbox
    draw = ImageDraw.Draw(image)
    draw.rectangle(_xyxy(bbox), fill=(225, 225, 225))
    skin = rng.choice(((224, 177, 139), (205, 155, 115), (180, 130, 95)))
    hair = rng.choice(((35, 30, 25), (70, 45, 25), (20, 20, 20)))
    cx = x + width / 2
    head_radius = width * 0.24
    draw.ellipse(
        (cx - head_radius, y + height * 0.18, cx + head_radius, y + height * 0.58),
        fill=skin,
    )
    draw.arc(
        (cx - head_radius - 2, y + height * 0.12, cx + head_radius + 2, y + height * 0.42),
        180,
        360,
        fill=hair,
        width=8,
    )
    eye_y = y + height * 0.36
    draw.ellipse((cx - head_radius * 0.45, eye_y, cx - head_radius * 0.25, eye_y + 4), fill=BLACK)
    draw.ellipse((cx + head_radius * 0.25, eye_y, cx + head_radius * 0.45, eye_y + 4), fill=BLACK)
    draw.arc(
        (cx - head_radius * 0.3, eye_y + 10, cx + head_radius * 0.3, eye_y + 22),
        0,
        180,
        fill=(120, 60, 50),
        width=2,
    )
    draw.ellipse(
        (x + width * 0.15, y + height * 0.65, x + width * 0.85, y + height * 1.15),
        fill=(60, 80, 120),
    )


def _new_mask(size: tuple[int, int]) -> Image.Image:
    return Image.new("L", size, 0)


def _stamp_anchor(parent: DocumentRecord, page_size: tuple[int, int]) -> tuple[int, int]:
    page_width, page_height = page_size
    for key in _STAMP_KEYS:
        if key in parent.field_boxes:
            x, y, width, _ = parent.field_boxes[key]
            return int(x + width + 55), int(y - 10)
    return page_width - 150, page_height - 160


def _identity_from_record(record: DocumentRecord) -> Identity:
    fields = record.fields
    return Identity(
        source_id=record.source_id,
        nombres=str(fields.get("nombres", "Nombre")),
        apellidos=str(fields.get("apellidos", "Apellido")),
        dni=str(fields.get("dni", fields.get("numero", "00000000"))),
        birth_date=_parse_date(str(fields.get("fecha_nacimiento", "01/01/1990"))),
        sexo=str(fields.get("sexo", "M")),
        estado_civil=str(fields.get("estado_civil", "Soltero(a)")),
        direccion=str(fields.get("direccion", "")),
        telefono="900000000",
        program=str(fields.get("programa", "")),
        module=str(fields.get("modulo", "")),
        period=str(fields.get("periodo", "")),
        admission_date=_parse_date(str(fields.get("fecha_matricula", "01/03/2024"))),
        issue_date=_parse_date(str(fields.get("fecha_emision", "01/01/2025"))),
        resolution=str(fields.get("resolucion", "")),
    )


def _parse_date(value: str) -> Any:
    from datetime import date, datetime

    try:
        return datetime.strptime(value, "%d/%m/%Y").date()
    except ValueError:
        return date(1990, 1, 1)


def write_attacks_manifest(
    records: list[DocumentRecord],
    out_dir: Path,
    *,
    seed: int = 42,
    copies: int = 1,
    per_attack: Mapping[str, int] | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "attacks_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "generator": "fraude.synth.attacks",
        "count": len(records),
        "seed": seed,
        "copies": copies,
        "by_attack": _count(record.primary_attack for record in records),
        "by_document_type": _count(record.document_type for record in records),
    }
    if per_attack is not None:
        summary["per_attack"] = dict(sorted(per_attack.items()))
    (out_dir / "attacks_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> list[DocumentRecord]:
    """Carga un manifiesto JSONL de auténticos y reconstruye los dataclasses."""
    records: list[DocumentRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        data["attack_types"] = tuple(data.get("attack_types", ()))
        data["mask_paths"] = tuple(data.get("mask_paths", ()))
        data["page_size"] = tuple(data.get("page_size", (0, 0)))
        data["field_boxes"] = {
            key: tuple(value) for key, value in data.get("field_boxes", {}).items()
        }
        data["regions"] = tuple(
            Region(
                attack_type=region["attack_type"],
                bbox=tuple(region["bbox"]),
                field=region.get("field"),
                original_value=region.get("original_value"),
                altered_value=region.get("altered_value"),
                mask_path=region.get("mask_path"),
            )
            for region in data.get("regions", ())
        )
        records.append(DocumentRecord(**data))
    return records


def _count(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera manipulaciones sintéticas T1-T8.")
    parser.add_argument("--data", type=Path, required=True, help="Directorio con manifest.jsonl.")
    parser.add_argument("--out", type=Path, default=None, help="Directorio de salida.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible.")
    parser.add_argument("--copies", type=int, default=1, help="Muestras por tipo de ataque.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI para los re-render T8.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data)
    out_dir = Path(args.out) if args.out is not None else data_dir
    authentic = load_manifest(data_dir / "manifest.jsonl")
    records = generate_attacks(
        authentic,
        out_dir,
        source_dir=data_dir,
        seed=args.seed,
        copies=args.copies,
        dpi=args.dpi,
    )
    print(f"Generadas {len(records)} muestras manipuladas en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
