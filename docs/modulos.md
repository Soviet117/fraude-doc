# Módulos del sistema

El prototipo es un monolito Python con Streamlit + SQLite. No hay
microservicios, colas ni servicios externos. Cada paquete tiene una
responsabilidad y pruebas en `tests/`.

## 1. Mapa rápido

| Paquete | Responsabilidad | Archivos clave | Fase |
|---|---|---|---|
| `core/` | Contrato `Observation`, configuración y logging | `schemas.py`, `config.py` | F0 |
| `ingest/` | Hash SHA-256, deduplicación e inventario | `hashing.py`, `inventory.py` | F1 |
| `preprocess/` | Calidad, tipo de captura, DPI estimado | `quality.py` | F2 |
| `synth/` | Plantillas, ataques T1–T8, aumentación y folds | `templates.py`, `attacks.py`, `augment.py`, `folds.py` | F1–F2 |
| `forensics/` | ELA, ruido, cromática, JPEG, metadatos, copy-move | `classic.py` | F3 |
| `ocr/` | docTR, texto y layout | `engine.py`, `layout.py` | F3 |
| `extract/` | Campos por tipo documental | `fields.py` | F3 |
| `rules/` | Reglas semánticas con severidad | `engine.py` | F3 |
| `registry/` | Cruce con el espejo del CETPRO | `cross.py`, `build.py` | F3 |
| `nlp/` | F6: normalización, semántica, duplicados y explicación | `semantic.py`, `duplicates.py`, `explain.py` | F6 |
| `models/` | F4 (EfficientNet-B0) y F5 (U-Net) | `train.py`, `seg_train.py`, `infer.py` | F4–F5 |
| `fusion/` | F7: features, calibración, entrenamiento, SHAP y barrido de semillas | `features.py`, `calibration.py`, `train.py`, `infer.py` | F7 |
| `pipeline/` | Orquestación F3 y pipeline integral F3+F4+F5+F6+F7 | `run.py`, `full.py` | F3/F7 |
| `eval/` | Métricas, línea base, evaluación de muestras y corpus v2 | `metrics.py`, `baseline.py`, `v2_pipeline.py` | F3–F8 |
| `report/` | Informe PDF | `pdf.py` | F9 |
| `store/` | SQLite: documentos, análisis y feedback | `history.py` | F9 |
| `apps/demo/` | Interfaz Streamlit | `app.py` | F9 |

## 2. Flujo de dependencias

```
apps/demo
   └── pipeline/full
          ├── preprocess → forensics → ocr → extract → rules → registry
          ├── nlp/semantic (F6) + nlp/duplicates
          ├── models (F4/F5) + fusion (F7)
          ├── report/pdf
          └── store/history (SQLite)
```

La demo solo importa `pipeline/full.py`; no conoce los detalles de cada modelo.
Los mapas y máscaras se guardan por ruta, no se embeben en el JSON.

## 3. Detalle por grupo

### Datos y síntesis

- **core/schemas.py**: `Observation`, `SourceInfo`, `QualityReport`,
  `ForensicsReport`, `RuleFinding`, `RegistryReport`, `FusionReport`,
  `Evidence`. Contrato congelado; los campos nuevos deben ser opcionales.
- **ingest/hashing.py**: SHA-256 de archivo y bytes; deduplicación exacta.
- **preprocess/quality.py**: resolución mínima, desenfoque, inclinación,
  contraste, exposición y tipo de captura.
- **synth/**: genera PDF/PNG sintéticos, ataques T1–T8 con máscaras, cinco
  aumentaciones de captura y folds agrupados. Todo reproducible por semilla.

### Señales

- **forensics/classic.py**: ELA (solo JPEG), residuo de ruido, cromática,
  metadatos, JPEG ghosts y copy-move.
- **ocr/engine.py**: docTR; devuelve palabras, confianza y coordenadas.
- **extract/fields.py**: esquema por tipo documental; regex + etiquetas +
  validaciones. No usa LLM.
- **rules/engine.py**: fechas, DNI, promedios, horas, correlativos, calidad,
  metadatos y registro. Cada regla tiene código, severidad y mensaje.
- **registry/cross.py**: coincidencia difusa nombre/DNI contra el espejo
  sintético; devuelve `matched`, `score` y `diffs`.

### Inteligencia

- **nlp/semantic.py**: consistencia de nombre/DNI, fecha de emisión plausible y
  reutilización de documentos.
- **nlp/duplicates.py**: hash de píxeles + dHash + similitud de campos.
- **nlp/explain.py**: plantilla determinista y Ollama local opcional. El LLM
  solo redacta; si falla, se usa la plantilla.
- **models/model.py**: EfficientNet-B0 de página (F4).
- **models/unet.py**: U-Net de localización (F5).
- **fusion/features.py**: 62 características en cinco grupos (rules, forensics,
  visual, localization, semantic).
- **fusion/calibration.py**: identidad, isotónica y Platt; se elige por Brier.
- **fusion/infer.py**: score, banda y evidencias SHAP por muestra.

### Evaluación y salida

- **eval/metrics.py**: precision, recall, F1, FPR, PR-AUC, ROC-AUC, ECE,
  Brier, bootstrap.
- **eval/baseline.py**: score heurístico de F3.
- **eval/v2_pipeline.py**: ejecuta F3 sobre los 50 documentos base.
- **report/pdf.py**: informe con banda, evidencias, campos, explicación y mapa.
- **store/history.py**: SQLite con `documents`, `analyses` y `feedback`.
- **apps/demo/app.py**: pestañas de análisis integral, F3, F4, historial y fases.

## 4. Pruebas

`tests/` contiene 254 casos recogidos por pytest. Los archivos siguen el nombre
del módulo: `test_schemas.py`, `test_forensics.py`, `test_fusion.py`,
`test_history.py`, `test_report.py`, etc. La matriz visual está en
`docs/figuras/casos_prueba.png`.

## 5. Lo que no hay

- No hay FastAPI, Postgres, Redis, Kubernetes ni microservicios (ADR-10).
- No hay integración con RENIEC/MINEDU ni blockchain.
- No hay entrenamiento con documentos reales; todo el corpus es sintético.
- No hay decisión automática: la salida siempre es apoyo a revisión humana.
