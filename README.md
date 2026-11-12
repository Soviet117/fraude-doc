# Sistema de detección de fraude documental — CETPRO San Agustín

Prototipo de investigación aplicada que analiza documentos académicos (DNI, certificados de estudios, partidas, fichas de matrícula, actas, certificados de módulo y títulos de Auxiliar Técnico) combinando **visión por computadora** y **procesamiento de lenguaje natural**, y devuelve un **nivel de riesgo de autenticidad con evidencias explicables**.

> **Estado (2026-11-10):** F0–F7 cerradas en el entorno local; F8 parcial; F9 en curso. El sistema ya ejecuta el pipeline integral **F3 (calidad, forense, OCR, reglas, registro) + F4 (página) + F5 (localización) + F6 (semántica y duplicados) + F7 (fusión calibrada)** en una demo Streamlit con historial SQLite y exportación PDF/JSON. Sigue sin haber documentos reales del CETPRO: todos los números son una prueba de viabilidad sobre datos sintéticos, no una estimación de desempeño institucional.

## Resultados principales (corpus sintético v2)

| Rama | Resultado | Límite |
|---|---|---|
| F3 `stripped` | F1 0.200, FPR 0.083 (20 muestras v1) | Detecta T7/T8; no T1–T6 |
| F4 página | PR-AUC 0.712, F1 0.541, FPR 0.080 (270 OOF) | Detecta T5/T6; falla T1–T4 |
| F5 localización | Pixel-F1 calibrado 0.915 (270 OOF) | Dominado por T5/T6; T1–T4 ≈ 0 |
| F6 extracción | Tipo documental 0.960 (50 docs) | `director`/`turno` fallan; T7 trivial |
| F7 fusión | PR-AUC 0.980, F1 0.898, FPR 0.080, ECE 0.070 (50 OOF) | n pequeño, semilla principal 42, atajo T7 |

## Documentación

| Archivo | Para qué |
|---|---|
| [`plan-sistema-fraude-documental.md`](plan-sistema-fraude-documental.md) | Plan completo: stack, arquitectura, pipeline, fases, backlog, evaluación y riesgos |
| [`AGENTS.md`](AGENTS.md) | Contexto y reglas para agentes de IA que trabajen en el repo |
| [`MANUAL_USUARIO.txt`](MANUAL_USUARIO.txt) | Manual de usuario paso a paso, en texto plano |
| [`docs/manual_demo.md`](docs/manual_demo.md) | Arranque, guion de sustentación, cifras y solución de problemas |
| [`docs/modulos.md`](docs/modulos.md) | Detalle módulo por módulo y qué prueba cada uno |
| [`docs/diagramas.md`](docs/diagramas.md) | Casos de uso, arquitectura, ER SQLite y flujo del pipeline |
| [`docs/cap4_resultados.md`](docs/cap4_resultados.md) | Tabla maestra y figuras para el Cap. IV |
| [`docs/verificacion_final.md`](docs/verificacion_final.md) | Checklist de cierre, comandos y evidencia de la verificación |
| [`docs/f3_linea_base.md`](docs/f3_linea_base.md) | Línea base F3, umbrales e IC bootstrap |
| [`docs/f4_corpus_v2.md`](docs/f4_corpus_v2.md) | Data card del corpus v2 y resultados F4 |
| [`docs/f5_localizacion.md`](docs/f5_localizacion.md) | F5: U-Net, Pixel-F1, límites por ataque |
| [`docs/f6_semantica.md`](docs/f6_semantica.md) | F6: extracción, duplicados, explicación y fallback |
| [`docs/f7_fusion.md`](docs/f7_fusion.md) | F7: features, calibración, ablaciones y SHAP |
| [`docs/model_card_f4.md`](docs/model_card_f4.md) | Model card de F4 |
| [`docs/model_card_f7.md`](docs/model_card_f7.md) | Model card de F7 |
| [`docs/presentacion.md`](docs/presentacion.md) | Guion breve y cifras citables |
| [`docs/auditoria_muestras.md`](docs/auditoria_muestras.md) | Integridad y límites del corpus v1 |
| [`docs/ocr_muestras.md`](docs/ocr_muestras.md) | Medición de OCR real con docTR |
| [`docs/muestras_pipeline.md`](docs/muestras_pipeline.md) | Evaluación end-to-end del corpus v1 |

## Módulos y base de datos

| Grupo | Paquetes | Qué hace |
|---|---|---|
| Contrato y datos | `core/`, `ingest/`, `preprocess/`, `synth/` | `Observation`, hashes, calidad y corpus sintético |
| Señales | `forensics/`, `ocr/`, `extract/`, `rules/`, `registry/` | ELA/ruido, docTR, campos, reglas y registro |
| Inteligencia | `nlp/`, `models/`, `fusion/` | F6 semántica, F4/F5 y F7 calibrado |
| Salida | `eval/`, `report/`, `store/` | Métricas, PDF y SQLite |
| Presentación | `apps/demo/`, `pipeline/` | Streamlit y orquestación integral |

**Base de datos:** sí. Es SQLite, archivo `artifacts/history.db`, con tres
tablas: `documents` (documento y hash), `analyses` (resultado, score, banda,
payload JSON) y `feedback` (revisión humana). El esquema físico y los datos de
ejemplo están en [docs/diagramas.md](docs/diagramas.md).

**¿Puedo subir documentos?** Sí: imágenes PNG/JPG/TIF/BMP/WEBP desde la vista
*Análisis*. El sistema devuelve banda, evidencias, mapa F5, explicación y
PDF/JSON. No acepta PDF directamente (exportar la página a PNG).
No detecta "todo" con la misma calidad: T5/T6 son fuertes, T1–T4 son el límite
declarado y T7 del corpus es una copia exacta.

## Resumen técnico

- **Datos:** corpus sintético v2 (`data/synthetic/v2`, 50 documentos base, 25/25, 10 personas, 7 tipos, 300 imágenes) y corpus v1 (`data/muestras`, 20 muestras). DocTamper queda como trabajo futuro.
- **Pipeline:** calidad → tipo de captura → forense (ELA, ruido, cromática, metadatos, copy-move) → OCR docTR → extracción → reglas → registro → F6 semántica → F4/F5 → F7 (LightGBM + calibración) → explicación → SQLite/PDF.
- **Stack:** Python 3.13, PyTorch 2.14, docTR, spaCy, LightGBM, Streamlit, SQLite, reportlab, Docker.
- **Demo:** navegación lateral (Análisis, Historial, Ayuda), tema claro/oscuro, tooltips contextuales, subida de archivo o selección del corpus, descarga PDF/JSON y revisión humana en SQLite.
- **Infraestructura:** una imagen Docker idéntica en 3 PCs; el laboratorio (Ubuntu + GPU) se usa por SSH solo para entrenamiento.

## Principios

1. Ningún resultado sin reproducibilidad.
2. Ninguna decisión automática: el sistema prioriza revisión humana.
3. Ninguna señal sin explicación.
4. Ningún dato personal fuera del equipo.
5. Ningún modelo sin calibración.

## Arranque rápido

```bash
make setup   # crea .env con UID/GID
make up      # levanta el contenedor de desarrollo
make test    # pytest dentro del contenedor
make demo    # Streamlit en localhost:8501
```

Sin Docker:

```bash
uv sync --extra ml
uv run pytest -q
uv run ruff check .
uv run streamlit run apps/demo/app.py
```

## Comandos de datos y modelos

```bash
make synth-v2        # corpus v2: 50 documentos base + 250 aumentados
make validate-v2     # integridad, hashes, balance y particiones
make folds-v2        # folds agrupados de 5 vías
make pipeline-v2     # F3 sobre los 50 documentos base (OCR real)
make eval-f6         # extracción, duplicados y explicaciones
make features-f7     # tabla de 62 características para F7
make train-f7        # fusión + calibración + ablaciones
make figures-f7      # ROC/PR, confusión, calibración, umbrales, SHAP
make seeds-f7        # barrido de 5 semillas (F8 parcial)
make eval            # reproduce tablas y figuras OOF de F4-F7
make eval-system     # sanity check del pipeline integral sobre los 50 docs v2
make train-f4 ARGS="--epochs 12 --device cpu"
make eval-f4
make figures-f4
make gradcam-f4
make train-f5 ARGS="--epochs 12 --device cpu --image-size 128"
make eval-f5
make figures-f5
make diagrams        # casos de uso, arquitectura, ER y flujo
```

## Limitaciones declaradas

- **Sin documentos reales del CETPRO.** Ley N.° 29733 y alcance del curso; el corpus es sintético y el sistema no valida oficialmente documentos de RENIEC/MINEDU.
- **n = 50 documentos base.** Los intervalos son anchos y la varianza entre folds es alta; hay un barrido de 5 semillas para F7, pero F8 debe añadir robustez y kappa.
- **F5 no localiza T1–T4.** A 128 px las ediciones pequeñas se pierden; la fusión compensa con reglas y semántica, no con localización fina.
- **T7 trivial.** La copia exacta del corpus infla la detección de duplicados; no mide reutilización real.
- **Calibración identidad.** Con n pequeño, isotónica y Platt empeoran Brier/ECE; se elige identidad y se declara (ECE 0.070).
- **Sin despliegue institucional, sin kappa y sin verificación en las 3 PCs todavía.**
