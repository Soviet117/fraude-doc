# Cap. IV — Resultados experimentales (borrador reproducible)

Este documento concentra las tablas y figuras que alimentan el Capítulo IV.
Todos los números salen de comandos del repositorio y de artefactos en
`artifacts/`; ninguno proviene de una celda de notebook. El corpus es
**sintético** y no hay validación con documentos reales del CETPRO.

## 1. Tabla maestra

| Rama | Qué mide | Corpus / n | Precision | Recall | F1 | FPR | Otra métrica |
|---|---|---|---:|---:|---:|---:|---|
| F3 `stripped` | Heurístico + reglas + registro | 20 muestras v1 | 0.500 | 0.125 | 0.200 | 0.083 | — |
| F4 página | Manipulación visual de página | 270 OOF v2 | 0.803 | 0.408 | 0.541 | 0.080 | PR-AUC 0.712 |
| F5 localización | Píxeles de la región alterada | 270 OOF v2 | 0.907* | 0.924* | 0.915* | — | Pixel-AUC 0.993 |
| F6 extracción | Tipo documental y campos | 50 docs v2 | — | — | — | — | Tipo 0.960 |
| F6 duplicados | T7 en corpus | 50 docs v2 | — | 1.000 | — | 0.000 | Copia exacta |
| F7 fusión | Riesgo documental | 50 OOF v2 | 0.917 | 0.880 | 0.898 | 0.080 | PR-AUC 0.980 |

\* F5 calibrado, dominado por T5/T6. En T1–T4 el Pixel-F1 es ≈ 0. La
comparación entre ramas **no es controlada**: distinto corpus, tamaño y
prevalencia.

## 2. F3 — línea base heurística

| Vista | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|
| `fields_source=manifest_oracle` | 0.833 | 0.417 | 0.556 | 0.125 |
| `stripped` (OCR real) | 0.500 | 0.125 | 0.200 | 0.083 |

La tabla oráculo aísla reglas y registro; **no** es el desempeño del pipeline
completo. Con OCR real, F3 detecta T7/T8 y no T1–T6. Detalle en
[`f3_linea_base.md`](f3_linea_base.md) y [`muestras_pipeline.md`](muestras_pipeline.md).

## 3. F4 — clasificador de página

| Métrica | Valor | IC 95 % |
|---|---:|---|
| PR-AUC | 0.712 | 0.641–0.778 |
| ROC-AUC | 0.653 | 0.581–0.721 |
| F1 @0.5 | 0.541 | 0.446–0.625 |
| Precision @0.5 | 0.803 | — |
| Recall @0.5 | 0.408 | — |
| FPR @0.5 | 0.080 | 0.040–0.124 |

| Ataque | n | Recall @0.5 |
|---|---:|---:|
| T1 | 4 | 0.000 |
| T2 | 4 | 0.250 |
| T3 | 3 | 0.333 |
| T4 | 3 | 0.000 |
| T5 | 3 | 1.000 |
| T6 | 3 | 1.000 |

Figuras: [`f4_training_curves.png`](figuras/f4_training_curves.png),
[`f4_confusion_matrix.png`](figuras/f4_confusion_matrix.png),
[`f4_attack_recall.png`](figuras/f4_attack_recall.png),
[`f4_calibration.png`](figuras/f4_calibration.png). Detalle en
[`f4_corpus_v2.md`](f4_corpus_v2.md) y [`model_card_f4.md`](model_card_f4.md).

## 4. F5 — localización

| Escenario | Pixel-F1 | IC 95 % | IoU |
|---|---:|---|---:|
| Fijo 0.5 | 0.8456 | 0.7474–0.8945 | 0.7325 |
| Calibrado | 0.9152 | 0.8701–0.9423 | 0.8436 |
| Oráculo | 0.9505 | — | 0.9057 |

| Ataque | Pixel-F1 @0.5 | Pixel-F1 calibrado |
|---|---:|---:|
| T1 | 0.023 | 0.023 |
| T2 | 0.017 | 0.016 |
| T3 | 0.123 | 0.002 |
| T4 | 0.037 | 0.031 |
| T5 | 0.563 | 0.799 |
| T6 | 0.992 | 0.971 |

Figuras: [`f5_pixel_f1_attack.png`](figuras/f5_pixel_f1_attack.png),
[`f5_threshold_sweep.png`](figuras/f5_threshold_sweep.png),
[`f5_qualitative.png`](figuras/f5_qualitative.png). Detalle en
[`f5_localizacion.md`](f5_localizacion.md).

**Lectura:** el promedio alto está dominado por T5/T6; T1–T4 siguen sin
localización fiable. La causa es la resolución (128 px), no el umbral.

## 5. F6 — semántica y extracción

| Métrica | Valor |
|---|---:|
| Tipo documental | 0.960 |
| Campos con exactitud ≥ 0.9 | mayoría de fechas, nombres, DNI, periodo |
| Campos que fallan | `director` 0.000, `turno` 0.000, foto/sello (no texto) |
| Recall T7 | 1.000 (copia exacta) |
| FPR auténticos (T7) | 0.000 |
| Explicaciones no vacías | 50/50 |
| Fuente de explicación | plantilla (LLM apagado) |

Detalle en [`f6_semantica.md`](f6_semantica.md).

## 6. F7 — fusión calibrada

| Métrica | Valor | IC 95 % |
|---|---:|---|
| PR-AUC | 0.9796 | 0.9458–1.0000 |
| ROC-AUC | 0.9760 | — |
| F1 @0.5 | 0.8980 | 0.8077–0.9715 |
| Precision @0.5 | 0.9167 | — |
| Recall @0.5 | 0.8800 | — |
| FPR @0.5 | 0.0800 | — |
| Brier (identidad) | 0.0854 | — |
| ECE (identidad) | 0.0700 | — |

### Ablaciones acumulativas

| Grupo | PR-AUC | F1 @0.5 | FPR @0.5 |
|---|---:|---:|---:|
| rules | 0.6352 | 0.5778 | 0.2800 |
| + forensics | 0.7335 | 0.7111 | 0.1600 |
| + visual (F4) | 0.7204 | 0.7083 | 0.2400 |
| + localization (F5) | 0.7678 | 0.8400 | 0.1600 |
| + semantic (F6) | 0.9796 | 0.8980 | 0.0800 |

Figuras: [`f7_roc_pr.png`](figuras/f7_roc_pr.png),
[`f7_confusion.png`](figuras/f7_confusion.png),
[`f7_calibration.png`](figuras/f7_calibration.png),
[`f7_threshold_sweep.png`](figuras/f7_threshold_sweep.png),
[`f7_ablations.png`](figuras/f7_ablations.png),
[`f7_importance.png`](figuras/f7_importance.png). Detalle en
[`f7_fusion.md`](f7_fusion.md) y [`model_card_f7.md`](model_card_f7.md).

## 6.1 Sanity check de integración (no es validación)

Sobre los **50 documentos base**, el pipeline integral con F7 asignó:

| Etiqueta | Riesgo bajo | Riesgo medio | Riesgo alto |
|---|---:|---:|---:|
| Auténticos (25) | 25 | 0 | 0 |
| Manipulados (25) | 0 | 1 | 24 |

El único manipulado en riesgo medio es `partida_nacimiento_0002_t1_c2`
(T1, p = 0.340), cerca del umbral de 0.33. Es un chequeo **in-sample**: el
modelo final de F7 se entrenó con esas filas y los checkpoints de F4/F5 son los
de demo. Sirve para verificar que el cableado funciona y que la demo no produce
falsos positivos en el corpus, **no** como métrica de generalización. El
resultado OOF de la sección 6 es el que se cita.

### 6.2 Barrido de semillas (F8 parcial)

| Métrica | Media ± desv. (5 semillas) |
|---|---|
| PR-AUC | 0.9800 ± 0.0010 |
| ROC-AUC | 0.9770 ± 0.0014 |
| F1 @0.5 | 0.8980 ± 0.0000 |
| Precision | 0.9167 ± 0.0000 |
| Recall | 0.8800 ± 0.0000 |
| FPR | 0.0800 ± 0.0000 |

Las semillas cambian el modelo, pero no el punto de operación en este corpus.
No sustituye la evaluación de robustez ni el kappa de F8.

## 7. Sistema y demo

| Métrica | Valor (CPU, 16 núcleos) |
|---|---:|
| Latencia total p50 | ≈ 3.3 s (media 3.8 s; primer análisis ≈ 21 s) |
| OCR | ≈ 2.0–3.1 s |
| Forense | ≈ 0.4 s |
| F4 + F5 | ≈ 0.1 s |
| F7 | ≈ 0.003 s |
| PDF | < 0.2 s |

La demo permite análisis integral, descarga PDF/JSON, historial SQLite y
revisión humana. Diagramas en [`diagramas.md`](diagramas.md):
[`casos_uso.png`](figuras/casos_uso.png),
[`arquitectura.png`](figuras/arquitectura.png),
[`er_historial.png`](figuras/er_historial.png),
[`flujo_pipeline.png`](figuras/flujo_pipeline.png).

## 8. Discusión

- **El valor está en la fusión, no en una señal aislada.** F4 sola falla T1–T4;
  F5 sola falla T1–T4; F6 aporta semántica y duplicados; F7 combina y mejora el
  F1 de todas las ramas.
- **La calibración es un problema real.** Con n = 50, isotónica y Platt
  empeoran ECE; el protocolo elige identidad y lo declara. Esto es coherente con
  el hallazgo de DocForge-Bench: un AUC alto no garantiza un umbral útil.
- **La localización fina sigue abierta.** El resultado de F5 es honesto: sirve
  para regiones grandes, no para ediciones pequeñas.
- **El T7 es trivial.** El salto de la ablación semántica está inflado por una
  copia exacta; no debe citarse como detección de reutilización real.

## 9. Limitaciones

1. Todo el corpus es sintético; no hay documentos reales del CETPRO.
2. 50 documentos base y 10 personas; IC anchos. Semilla principal 42 y barrido de 5 semillas (PR-AUC 0.9800 ± 0.0010), pero faltan robustez y kappa.
3. Sin validación de RENIEC/MINEDU ni despliegue institucional.
4. F5 no localiza T1–T4; F4 no ve ediciones pequeñas.
5. F7 puede aprender atajos del generador (metadatos, plantillas, duplicado).
6. El LLM no se evaluó con jueces humanos; la plantilla es el fallback probado.
7. F8 (varias semillas, robustez, McNemar, kappa) y la verificación en las 3 PCs
   siguen pendientes.

## 10. Mapeo con los objetivos específicos

| Objetivo | Evidencia |
|---|---|
| OE1 Recopilar documentos | Corpus v2, data card, taxonomía T1–T8 |
| OE2 Preparar documentos | Pipeline de calidad, normalización, generador y aumentación |
| OE3 Diseñar el modelo | F3–F7, arquitectura híbrida, ablaciones |
| OE4 Desarrollar el sistema | Demo Streamlit, SQLite, PDF, diagramas |
| OE5 Evaluar el desempeño | Tabla maestra, IC bootstrap, calibración, límites |
