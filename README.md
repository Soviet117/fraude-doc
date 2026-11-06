# Sistema de detección de fraude documental — CETPRO San Agustín

Prototipo de investigación aplicada que analiza documentos académicos (DNI, certificados de estudios, partidas, fichas de matrícula, actas, certificados de módulo y títulos de Auxiliar Técnico) combinando **visión por computadora** y **procesamiento de lenguaje natural**, y devuelve un **nivel de riesgo de autenticidad con evidencias explicables**.

> Estado: **fase F3 (línea base) cerrada de forma provisional** — F0–F2 cerrados en el entorno local con datos sintéticos (entorno, contrato `Observation`, ingesta, calidad, dataset sintético v1, ataques T1–T8 y particiones agrupadas). El alcance operativo de este prototipo es el corpus generado por IA de `data/muestras`; **no se usarán documentos reales del CETPRO**. F3 integra forense clásico, OCR, extracción, reglas y cruce con registro, y se evaluó de extremo a extremo sobre las 20 muestras. docTR está instalado y el OCR real fue medido (tipo documental 1.000, extracción de campos 0.923), pero el corpus **no alcanza para entrenar F4**: hay una sola muestra por ataque. **F4 no se inicia** hasta ampliar el corpus sintético con más personas y más muestras por ataque; la falta de documentos reales ya no es el bloqueo.

## Documentación

| Archivo | Para qué |
|---|---|
| [`plan-sistema-fraude-documental.md`](plan-sistema-fraude-documental.md) | Plan completo: stack, arquitectura, pipeline, fases, backlog, evaluación y riesgos |
| [`AGENTS.md`](AGENTS.md) | Contexto y reglas para agentes de IA que trabajen en el repo |
| [`docs/guia_anotacion.md`](docs/guia_anotacion.md) | Taxonomía T1–T8 y criterios de anotación |
| [`docs/solicitud_cetpro.md`](docs/solicitud_cetpro.md) | Plantilla de solicitud (no forma parte del alcance actual) |
| [`docs/f3_linea_base.md`](docs/f3_linea_base.md) | Tabla F3 provisional, umbrales, IC bootstrap y limitaciones |
| [`docs/auditoria_muestras.md`](docs/auditoria_muestras.md) | Composición, integridad y límites del corpus local de 20 muestras |
| [`docs/ocr_muestras.md`](docs/ocr_muestras.md) | Medición de OCR real con docTR, resultados y errores residuales |
| [`docs/muestras_pipeline.md`](docs/muestras_pipeline.md) | Evaluación end-to-end de las 20 muestras, modos `as_is`/`stripped` y límites |

## Resumen técnico

- **Datos**: corpus sintético generado por IA (`data/muestras`) y dataset sintético v1, con etiquetas a nivel de píxel; DocTamper para preentrenamiento (trabajo futuro).
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

- **F0–F2 cerrados en el entorno local:** Docker, CI, contrato `Observation`, ingesta por hash, calidad, dataset sintético v1 (348 muestras), ataques T1–T8, aumentación y particiones agrupadas. F1 se cierra con el corpus sintético; no se incorporarán documentos reales.
- **F3 implementado:** forense clásico, adaptador docTR, extracción por tipo, motor de reglas, cruce con registro y pipeline integrado.
- **F3 evaluado de forma provisional:** tabla con `fields_source=manifest_oracle` en [`docs/f3_linea_base.md`](docs/f3_linea_base.md) para aislar reglas y registro, más evaluación end-to-end de las 20 muestras en [`docs/muestras_pipeline.md`](docs/muestras_pipeline.md). Con OCR real, F3 detecta T7 y T8, pero **no T1–T6**; la tabla oráculo no debe citarse como desempeño del pipeline completo.
- **OCR real medido:** docTR 1.1.0 sobre las 20 muestras; recall de campos 0.924, tipo documental 1.000 y extracción 0.923. Los cuatro fallos restantes son errores reales de OCR y se documentan sin maquillar en [`docs/ocr_muestras.md`](docs/ocr_muestras.md).
- **Corpus local auditado:** 20/20 hashes correctos, 0 errores de integridad y 20/20 imágenes aptas para calidad. Los metadatos de Photoshop separan 8/8 manipulados (atajo del generador, no detección) y copy-move no marca ninguna muestra. Detalle en [`docs/auditoria_muestras.md`](docs/auditoria_muestras.md).
- **Pendiente:** verificar la imagen en la PC del compañero y el laboratorio; ampliar el corpus sintético (más personas y muestras por ataque) antes de F4; conseguir evaluadores humanos para el kappa.

## Arranque rápido

```bash
make setup   # crea .env con UID/GID
make up      # levanta el contenedor de desarrollo
make test    # pytest dentro del contenedor
make demo    # Streamlit en localhost:8501
make audit-samples  # audita data/muestras con calidad/forense, sin OCR
make ocr-samples    # mide OCR real con docTR (requiere extra ml)
make eval-samples   # evalúa el pipeline completo en modos as_is y stripped
```

Sin Docker, para tests y lint: `uv sync && uv run pytest -q && uv run ruff check .`.
La tabla F3 se regenera con los comandos de [`docs/f3_linea_base.md`](docs/f3_linea_base.md).
