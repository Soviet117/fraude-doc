# Sistema de detección de fraude documental — CETPRO San Agustín

Prototipo de investigación aplicada que analiza documentos académicos (DNI, certificados de estudios, partidas, fichas de matrícula, actas, certificados de módulo y títulos de Auxiliar Técnico) combinando **visión por computadora** y **procesamiento de lenguaje natural**, y devuelve un **nivel de riesgo de autenticidad con evidencias explicables**.

> Estado: **F4 experimental cerrado (prueba de viabilidad)** — F0–F3 cerrados en el entorno local. El corpus sintético v2 tiene 50 documentos base (25 auténticos y 25 manipulados, 10 personas, 7 tipos) y 300 imágenes con aumentación de captura; las particiones fijas de F2/F3 se conservan y se añadió validación cruzada agrupada de 5 vías. F4 entrena un EfficientNet-B0 de página con Grad-CAM y evaluación calibrada. Resultado principal: PR-AUC OOF 0.712 (IC 0.641-0.778), F1@0.5 0.541 y FPR 0.080; detecta T5/T6 y falla T1/T4, como se esperaba de un clasificador de página. Sigue sin haber documentos reales del CETPRO: los números de F4 son una prueba de viabilidad sobre datos sintéticos, no una estimación de desempeño institucional. La localización formal (F5), el kappa y la fusión calibrada (F7) siguen fuera de este cierre.

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
| [`docs/f4_corpus_v2.md`](docs/f4_corpus_v2.md) | Corpus v2 de 50 documentos, folds agrupados, entrenamiento F4 y límites |
| [`docs/model_card_f4.md`](docs/model_card_f4.md) | Model card de F4: uso previsto, datos, métricas y limitaciones |
| [`docs/presentacion.md`](docs/presentacion.md) | Guion de demo, cifras citables, límites y checklist de sustentación |

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
- **Corpus v2 para F4:** 50 documentos base (25/25), 10 personas, 7 tipos y 5 aumentaciones por documento; 300 imágenes y manifiestos con hashes. Validación de integridad, balance y folds agrupados en [`docs/f4_corpus_v2.md`](docs/f4_corpus_v2.md).
- **F4 experimental cerrado:** EfficientNet-B0 de página, validación cruzada agrupada de 5 vías, OOF, calibración isotónica cruzada, umbrales 0.5/calibrado/FPR≤0.2/oráculo, IC bootstrap, desglose por ataque y Grad-CAM. Resultados OOF: PR-AUC 0.712 (IC 0.641-0.778), ROC-AUC 0.653, F1@0.5 0.541 (IC 0.446-0.625), precision 0.803, recall 0.408 y FPR 0.080. Detecta T5/T6 (recall 1.000) y falla T1/T4; los folds 0 y 4 quedan cerca del azar porque no contienen T5/T6. Supera el piso F3 en el corpus sintético (F1 0.541 vs 0.200), sin comparación controlada. Detalle y límites en [`docs/f4_corpus_v2.md`](docs/f4_corpus_v2.md). Comandos `make synth-v2`, `validate-v2`, `folds-v2`, `train-f4`, `eval-f4` y `gradcam-f4`.
- **Pendiente para la presentación:** F5 (localización/parches para T1–T4), F6 (rama semántica), F7 (fusión calibrada), F8 (evaluación integral con varias semillas) y F9 (demo final/Cap. IV). También queda verificar la imagen en la PC del compañero y el laboratorio, y conseguir evaluadores humanos para el kappa.

## Arranque rápido

```bash
make setup   # crea .env con UID/GID
make up      # levanta el contenedor de desarrollo
make test    # pytest dentro del contenedor
make demo    # Streamlit en localhost:8501
make audit-samples  # audita data/muestras con calidad/forense, sin OCR
make ocr-samples    # mide OCR real con docTR (requiere extra ml)
make eval-samples   # evalúa el pipeline completo en modos as_is y stripped
make synth-smoke    # lote de humo de 3 documentos con manifiestos
make synth-v2       # corpus v2 de 50 documentos base + 250 aumentados
make validate-v2    # valida hashes, balance y particiones del corpus v2
make folds-v2       # folds agrupados de 5 vías por persona
make train-f4       # entrena F4 (ARGS="--epochs 12 --device cpu")
make eval-f4        # métricas OOF, calibración y umbrales
make gradcam-f4     # mapas de calor de TP/TN/FP/FN
```

Sin Docker, para tests y lint: `uv sync && uv run pytest -q && uv run ruff check .`.
La tabla F3 se regenera con los comandos de [`docs/f3_linea_base.md`](docs/f3_linea_base.md).
