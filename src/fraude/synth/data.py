"""Datos ficticios deterministas para el generador sintético.

Todos los nombres, DNIs, direcciones y notas son inventados. No se usa ningún
dato real de estudiantes ni del CETPRO. Las fechas son fijas para que el
dataset sea reproducible con la misma semilla.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

FIRST_NAMES_MALE = (
    "José",
    "Luis",
    "Carlos",
    "Miguel",
    "Jorge",
    "Pedro",
    "Juan",
    "Diego",
    "Andrés",
    "Fernando",
    "Ricardo",
    "Alberto",
)

FIRST_NAMES_FEMALE = (
    "María",
    "Ana",
    "Rosa",
    "Carmen",
    "Lucía",
    "Elena",
    "Patricia",
    "Sofía",
    "Gabriela",
    "Milagros",
    "Ruth",
    "Yesenia",
)

LAST_NAMES = (
    "Quispe",
    "Mamani",
    "Flores",
    "Rojas",
    "Huamán",
    "Vásquez",
    "Chávez",
    "Sánchez",
    "Ramírez",
    "Torres",
    "Díaz",
    "Rodríguez",
    "García",
    "Pérez",
    "Castillo",
)

PROGRAMS: dict[str, tuple[str, ...]] = {
    "Computación e Informática": (
        "Ofimática y herramientas digitales",
        "Mantenimiento de equipos de cómputo",
        "Desarrollo de aplicaciones web",
    ),
    "Administración de Empresas": (
        "Gestión de procesos administrativos",
        "Contabilidad básica",
        "Atención al cliente",
    ),
    "Electrónica Industrial": (
        "Instalaciones eléctricas",
        "Electrónica básica",
        "Automatización industrial",
    ),
    "Mecánica Automotriz": (
        "Mantenimiento de motores",
        "Sistemas de frenos y suspensión",
        "Diagnóstico electrónico",
    ),
    "Enfermería Técnica": (
        "Primeros auxilios",
        "Cuidados básicos del paciente",
        "Bioseguridad hospitalaria",
    ),
    "Cocina": (
        "Técnicas culinarias básicas",
        "Costos y presupuestos de cocina",
        "Seguridad e higiene alimentaria",
    ),
}

ADDRESSES = (
    "Jr. Los Olivos 245, Nueva Cajamarca",
    "Av. San Martín 1180, Nueva Cajamarca",
    "Calle Amazonas 312, Rioja",
    "Jr. Bolognesi 560, Moyobamba",
    "Av. Perú 890, Nueva Cajamarca",
    "Calle Las Palmeras 147, Nueva Cajamarca",
)

PLACES = (
    "Nueva Cajamarca, San Martín",
    "Rioja, San Martín",
    "Moyobamba, San Martín",
    "Tarapoto, San Martín",
)

SECONDARY_SCHOOLS = (
    "I.E. San Agustín de Nueva Cajamarca",
    "I.E. Andrés Avelino Cáceres",
    "I.E. José Carlos Mariátegui",
    "I.E. Sagrado Corazón de Jesús",
)

TEACHERS = (
    "Ing. Rosa Elena Paredes",
    "Lic. Marco Antonio Salazar",
    "Ing. Lucía Fernanda Ríos",
    "Prof. Julio César Mendoza",
    "Lic. Ana María Chávez",
)

DIRECTORS = (
    "Mg. Héctor Raúl Campos",
    "Lic. Silvia Roxana Pinedo",
)

PERIODS = ("2023-II", "2024-I", "2024-II", "2025-I", "2025-II")

SHIFTS = ("Mañana", "Tarde", "Noche")


@dataclass(frozen=True)
class Identity:
    """Identidad ficticia compartida por todos los documentos de una persona."""

    source_id: str
    nombres: str
    apellidos: str
    dni: str
    birth_date: date
    sexo: str
    estado_civil: str
    direccion: str
    telefono: str
    program: str
    module: str
    period: str
    admission_date: date
    issue_date: date
    resolution: str

    @property
    def full_name(self) -> str:
        return f"{self.apellidos}, {self.nombres}"


def generate_identity(
    rng: random.Random,
    index: int,
    used_dnis: set[str],
) -> Identity:
    """Genera una identidad ficticia única de forma determinista."""
    female = rng.random() < 0.5
    first_pool = FIRST_NAMES_FEMALE if female else FIRST_NAMES_MALE
    nombres = f"{rng.choice(first_pool)} {rng.choice(first_pool)}"
    apellidos = f"{rng.choice(LAST_NAMES)} {rng.choice(LAST_NAMES)}"
    dni = _unique_dni(rng, used_dnis)

    program = rng.choice(tuple(PROGRAMS))
    module = rng.choice(PROGRAMS[program])
    birth_date = date(rng.randint(1982, 2004), rng.randint(1, 12), rng.randint(1, 28))
    admission_date = date(rng.randint(2022, 2025), rng.choice((3, 4, 8, 9)), rng.randint(1, 28))
    issue_date = admission_date + timedelta(days=rng.randint(300, 720))

    return Identity(
        source_id=f"persona_{index:04d}",
        nombres=nombres,
        apellidos=apellidos,
        dni=dni,
        birth_date=birth_date,
        sexo="F" if female else "M",
        estado_civil=rng.choice(("Soltero(a)", "Casado(a)", "Conviviente")),
        direccion=rng.choice(ADDRESSES),
        telefono=f"9{rng.randint(10_000_000, 99_999_999)}",
        program=program,
        module=module,
        period=rng.choice(PERIODS),
        admission_date=admission_date,
        issue_date=issue_date,
        resolution=f"RD-{rng.randint(1000, 9999)}-2025-CETPRO-SA",
    )


def build_fields(
    rng: random.Random,
    document_type: str,
    identity: Identity,
) -> dict[str, Any]:
    """Construye los campos visibles de un documento sintético."""
    if document_type == "dni":
        return {
            "numero": identity.dni,
            "nombres": identity.nombres,
            "apellidos": identity.apellidos,
            "fecha_nacimiento": format_date(identity.birth_date),
            "sexo": identity.sexo,
            "estado_civil": identity.estado_civil,
            "direccion": identity.direccion,
            "fecha_emision": format_date(identity.issue_date - timedelta(days=900)),
            "fecha_vencimiento": format_date(identity.issue_date + timedelta(days=1800)),
        }

    if document_type == "certificado_estudios":
        return {
            "estudiante": identity.full_name,
            "dni": identity.dni,
            "fecha_nacimiento": format_date(identity.birth_date),
            "institucion": rng.choice(SECONDARY_SCHOOLS),
            "anio_egreso": str(identity.birth_date.year + 17),
            "promedio": f"{rng.uniform(13.0, 19.0):.1f}",
            "director": rng.choice(DIRECTORS),
            "fecha_emision": format_date(identity.issue_date),
        }

    if document_type == "partida_nacimiento":
        return {
            "nombre": f"{identity.nombres} {identity.apellidos}",
            "fecha_nacimiento": format_date(identity.birth_date),
            "lugar_nacimiento": rng.choice(PLACES),
            "padre": _adult_name(rng, male=True),
            "madre": _adult_name(rng, male=False),
            "numero_acta": f"{rng.randint(100_000, 999_999)}",
            "fecha_registro": format_date(identity.birth_date + timedelta(days=rng.randint(5, 60))),
        }

    if document_type == "ficha_matricula":
        return {
            "estudiante": identity.full_name,
            "dni": identity.dni,
            "periodo": identity.period,
            "programa": identity.program,
            "modulo": identity.module,
            "turno": rng.choice(SHIFTS),
            "fecha_matricula": format_date(identity.admission_date),
            "sede": "Nueva Cajamarca",
        }

    if document_type == "acta_evaluacion":
        fields: dict[str, Any] = {
            "programa": identity.program,
            "modulo": identity.module,
            "docente": rng.choice(TEACHERS),
            "periodo": identity.period,
            "fecha_evaluacion": format_date(identity.issue_date),
            "director": rng.choice(DIRECTORS),
        }
        rows = [(identity.full_name, identity.dni, rng.randint(13, 19))]
        for _ in range(4):
            rows.append(
                (
                    _student_name(rng),
                    _unique_dni(rng, set()),
                    rng.randint(10, 19),
                )
            )
        for row_index, (name, dni, grade) in enumerate(rows):
            fields[f"estudiante_{row_index}"] = name
            fields[f"dni_{row_index}"] = dni
            fields[f"nota_{row_index}"] = str(grade)
        return fields

    if document_type == "certificado_modulo":
        return {
            "estudiante": identity.full_name,
            "dni": identity.dni,
            "programa": identity.program,
            "modulo": identity.module,
            "horas": str(rng.choice((120, 160, 200))),
            "nota": str(rng.randint(14, 19)),
            "fecha_emision": format_date(identity.issue_date),
            "director": rng.choice(DIRECTORS),
        }

    if document_type == "titulo_auxiliar_tecnico":
        return {
            "estudiante": identity.full_name,
            "dni": identity.dni,
            "programa": identity.program,
            "fecha_emision": format_date(identity.issue_date),
            "resolucion": identity.resolution,
            "director": rng.choice(DIRECTORS),
        }

    raise ValueError(f"Tipo de documento no soportado: {document_type}")


def format_date(value: date) -> str:
    """Formatea una fecha como ``dd/mm/aaaa``."""
    return value.strftime("%d/%m/%Y")


def _unique_dni(rng: random.Random, used_dnis: set[str]) -> str:
    while True:
        dni = f"{rng.randint(10_000_000, 99_999_999)}"
        if dni not in used_dnis:
            used_dnis.add(dni)
            return dni


def _adult_name(rng: random.Random, *, male: bool) -> str:
    pool = FIRST_NAMES_MALE if male else FIRST_NAMES_FEMALE
    return f"{rng.choice(pool)} {rng.choice(LAST_NAMES)}"


def _student_name(rng: random.Random) -> str:
    pool = FIRST_NAMES_FEMALE if rng.random() < 0.5 else FIRST_NAMES_MALE
    return f"{rng.choice(LAST_NAMES)}, {rng.choice(pool)} {rng.choice(pool)}"
