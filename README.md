# Sistema de detección de fraude documental — CETPRO San Agustín

Prototipo de investigación aplicada que analiza documentos académicos (DNI, certificados de estudios, partidas, fichas de matrícula, actas, certificados de módulo y títulos de Auxiliar Técnico) combinando **visión por computadora** y **procesamiento de lenguaje natural**, y devuelve un **nivel de riesgo de autenticidad con evidencias explicables**.

> Estado: **fase F3 (línea base) cerrada de forma provisional** — F0–F2 cerrados en el entorno local con datos sintéticos (entorno, contrato `Observation`, ingesta, calidad, dataset sintético v1, ataques T1–T8 y particiones agrupadas); la entrega de documentos reales del CETPRO (F1) sigue pendiente. F3 integra forense clásico, OCR, extracción, reglas y cruce con registro, con una tabla provisional sobre datos sintéticos. docTR ya está instalado y el OCR real fue medido sobre las 20 muestras locales (tipo documental 1.000, extracción de campos 0.923), pero el corpus **no alcanza para entrenar F4**. **F4 no se inicia** hasta contar con documentos reales del CETPRO y un corpus con más personas y muestras por ataque.

## Documentación

| Archivo | Para qué |
|---|---|
| [`plan-sistema-fraude-documental.md`](plan-sistema-fraude-documental.md) | Plan completo: stack, arquitectura, pipeline, fases, backlog, evaluación y riesgos |
| [`AGENTS.md`](AGENTS.md) | Contexto y reglas para agentes de IA que trabajen en el repo |
| [`docs/guia_anotacion.md`](docs/guia_anotacion.md) | Taxonomía T1–T8 y criterios de anotación |
| [`docs/solicitud_cetpro.md`](docs/solicitud_cetpro.md) | Plantilla de solicitud formal de documentos al CETPRO |
| [`docs/f3_linea_base.md`](docs/f3_linea_base.md) | Tabla F3 provisional, umbrales, IC bootstrap y limitaciones |
| [`docs/auditoria_muestras.md`](docs/auditoria_muestras.md) | Composición, integridad y límites del corpus local de 20 muestras |
| [`docs/ocr_muestras.md`](docs/ocr_muestras.md) | Medición de OCR real con docTR, resultados y errores residuales |

## Resumen técnico

- **Datos**: documentos reales anonimizados + generador programático de documentos sintéticos auténticos y manipulados (etiquetas a nivel de píxel) + DocTamper para preentrenamiento.
- **Pipeline**: calidad → tipo de captura → forense (ELA, ruido, cromática, metadatos, copy-move) → OCR → extracción de campos → reglas → cruce con registro → fusión calibrada (LightGBM) → explicación.
- **Stack**: Python 3.13, PyTorch 2.14, docTR, spaCy, LightGBM, Streamlit, SQLite, Docker.
- **Infraestructura**: una imagen Docker idéntica en 3 PCs; el laboratorio (Ubuntu + GPU) se usa por SSH solo para entrenamiento.

## Principios

1. Ningún resultado sin reproducibilidad.
2. Ninguna decisión automática: el sistema prioriza revisión humana.
3. Ninguna señal sin explicación.
4. Ningún dato personal fuera del equipo.
5. Ningún modelo sin calibración.

## Estado actual

- **F0–F2 cerrados en el entorno local:** Docker, CI, contrato `Observation`, ingesta por hash, calidad, dataset sintético v1 (348 muestras), ataques T1–T8, aumentación y particiones agrupadas. F1 conserva pendiente la entrega de documentos reales del CETPRO.
- **F3 implementado:** forense clásico, adaptador docTR, extracción por tipo, motor de reglas, cruce con registro y pipeline integrado.
- **F3 evaluado de forma provisional:** 172 tests pasando; tabla en [`docs/f3_linea_base.md`](docs/f3_linea_base.md) con `fields_source=manifest_oracle` para aislar reglas y registro.
- **OCR real medido:** docTR 1.1.0 sobre las 20 muestras; recall de campos 0.924, tipo documental 1.000 y extracción 0.923. Los cuatro fallos restantes son errores reales de OCR y se documentan sin maquillar en [`docs/ocr_muestras.md`](docs/ocr_muestras.md).
- **Corpus local auditado:** 20/20 hashes correctos, 0 errores de integridad y 20/20 imágenes aptas para calidad. Los metadatos de Photoshop separan 8/8 manipulados (atajo del generador, no detección) y copy-move no marca ninguna muestra. Detalle en [`docs/auditoria_muestras.md`](docs/auditoria_muestras.md).
- **Pendiente:** verificar la imagen en la PC del compañero y el laboratorio; obtener documentos reales y registro institucional para F4.

## Arranque rápido

```bash
make setup   # crea .env con UID/GID
make up      # levanta el contenedor de desarrollo
make test    # pytest dentro del contenedor
make demo    # Streamlit en localhost:8501
make audit-samples  # audita data/muestras con calidad/forense, sin OCR
make ocr-samples    # mide OCR real con docTR (requiere extra ml)
```

Sin Docker, para tests y lint: `uv sync && uv run pytest -q && uv run ruff check .`.
La tabla F3 se regenera con los comandos de [`docs/f3_linea_base.md`](docs/f3_linea_base.md).
