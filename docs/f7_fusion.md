# F7 — Fusión y calibración

> **Estado:** cerrado experimental. LightGBM sobre las señales de F3–F6,
> validación cruzada agrupada por persona, calibración elegida por Brier/ECE,
> umbrales fijo/calibrado/oráculo, IC bootstrap, ablaciones y SHAP. Los
> resultados son sobre el corpus sintético v2 (50 documentos base); no son una
> estimación de desempeño institucional.

## 1. Qué entra en la fusión

`src/fraude/fusion/features.py` define 62 características en cinco grupos:

| Grupo | Señales |
|---|---|
| `rules` | calidad, captura, OCR, número y severidad de reglas, códigos de regla |
| `forensics` | ELA, ruido, cromática, JPEG, metadatos, copy-move |
| `visual` | score F4 y bandera de ausencia |
| `localization` | max/mean/área de F5 y bandera de ausencia |
| `semantic` | tipo documental, registro, campos, duplicado, banderas F6 |

Los valores ausentes se marcan con `*_missing` para que el modelo no los
confunda con un cero real.

## 2. Protocolo

- 5 folds agrupados por `source_id` (persona), sin fuga entre documentos del
  mismo titular.
- Predicciones out-of-fold (OOF) para las 50 filas; el modelo final se
  reentrena con todas las filas y se guarda con su calibrador.
- Calibración comparada de forma cruzada: **identidad**, **isotónica** y
  **Platt**. Se elige la de menor Brier; en empate, menor ECE.
- Umbrales: fijo 0.5, calibrado 0.5, calibrado max-F1, calibrado con FPR ≤ 0.2
  y oráculo (solo techo diagnóstico).
- IC bootstrap al 95 % para precision, recall, F1, FPR, PR-AUC y ROC-AUC.
- Ablaciones acumulativas por familia de señal y SHAP medio absoluto.

Comandos:

```bash
make features-f7
make train-f7
make figures-f7
```

## 3. Resultados OOF

| Métrica | Valor | IC 95 % |
|---|---:|---|
| Documentos | 50 | — |
| Prevalencia | 0.500 | — |
| PR-AUC | 0.9796 | 0.9458–1.0000 |
| ROC-AUC | 0.9760 | — |
| F1 @0.5 | 0.8980 | 0.8077–0.9715 |
| Precision @0.5 | 0.9167 | — |
| Recall @0.5 | 0.8800 | — |
| FPR @0.5 | 0.0800 | — |
| Accuracy @0.5 | 0.9000 | — |
| Matriz @0.5 | TP 22 · TN 23 · FP 2 · FN 3 | — |

### Umbrales

| Escenario | Umbral | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Fijo 0.5 | 0.50 | 0.9167 | 0.8800 | 0.8980 | 0.0800 |
| Calibrado @0.5 | 0.50 | 0.9167 | 0.8800 | 0.8980 | 0.0800 |
| Calibrado max-F1 | 0.05 | 0.8889 | 0.9600 | 0.9231 | 0.1200 |
| Calibrado FPR≤0.2 | 0.08 | 0.9231 | 0.9600 | 0.9412 | 0.0800 |
| Oráculo | 0.05 | 0.8889 | 0.9600 | 0.9231 | 0.1200 |

## 4. Calibración: por qué se eligió identidad

| Versión | Brier | ECE |
|---|---:|---:|
| Identidad (score crudo) | 0.0854 | **0.0700** |
| Isotónica cruzada | 0.0865 | 0.1023 |
| Platt cruzada | 0.1080 | 0.1379 |

Con 50 documentos, isotónica y Platt **empeoran** Brier y ECE. El protocolo
elige identidad y lo declara: el score crudo ya está razonablemente calibrado
(ECE 0.070 < 0.10), y no se aplica una transformación que solo añade ruido. La
demo muestra el score con su banda y el método elegido; no se presenta como
probabilidad institucional.

### 4.1 Barrido de semillas (F8 parcial)

| Métrica | Media ± desv. (5 semillas: 1, 7, 21, 42, 123) |
|---|---|
| PR-AUC | 0.9800 ± 0.0010 |
| ROC-AUC | 0.9770 ± 0.0014 |
| F1 @0.5 | 0.8980 ± 0.0000 |
| Precision | 0.9167 ± 0.0000 |
| Recall | 0.8800 ± 0.0000 |
| FPR | 0.0800 ± 0.0000 |

La variación de PR-AUC/ROC-AUC es pequeña; el punto de operación no cambia.
Esto mide estabilidad ante la semilla, no robustez ante degradaciones ni
documentos reales. Comando: `uv run python -m fraude.fusion.seed_sweep`.

## 5. Ablaciones acumulativas

| Grupo | Features | PR-AUC | F1 @0.5 | FPR @0.5 |
|---|---:|---:|---:|---:|
| rules | 38 | 0.6352 | 0.5778 | 0.2800 |
| + forensics | 44 | 0.7335 | 0.7111 | 0.1600 |
| + visual (F4) | 46 | 0.7204 | 0.7083 | 0.2400 |
| + localization (F5) | 50 | 0.7678 | 0.8400 | 0.1600 |
| + semantic (F6) | 62 | **0.9796** | **0.8980** | **0.0800** |

Lectura honesta:

- Las reglas solas ya separan bastante (PR-AUC 0.676).
- Forense clásico ayuda; F4 sola no mejora el PR-AUC en este corpus (baja a
  0.715) y F5 sí aporta sobre el conjunto anterior.
- El salto grande viene de F6, sobre todo por `f6_duplicate` en T7. Ese T7 es
  una copia exacta; el salto no debe interpretarse como capacidad general de
  detectar reutilización.

## 6. Importancia (SHAP medio absoluto)

| Feature | SHAP |
|---|---:|
| f6_duplicate | 3.0333 |
| n_rules | 0.6467 |
| forensics_ela | 0.3461 |
| f5_mean_prob | 0.2766 |
| registry_score | 0.2553 |
| f6_document_reuse | 0.2295 |
| blur | 0.1828 |
| rule_REGISTRO_DIFERENCIA | 0.1233 |
| ocr_log_words | 0.0891 |
| f5_area_fraction | 0.0818 |

La explicación de la demo usa SHAP por muestra (`pred_contrib`) para ordenar
las evidencias, no solo la importancia global. Si una señal no se puede
traducir a un mensaje comprensible, no se muestra.

## 7. Integración en el sistema

- `src/fraude/fusion/infer.py` carga el bundle (`model`, `calibrator`,
  `feature_names`, `calibration_method`) y devuelve `FusionReport` con
  `p_fake`, `risk` y evidencias.
- Bandas: `< 0.33` bajo, `0.33–0.66` medio, `≥ 0.66` alto.
- `src/fraude/pipeline/full.py` encadena F3 + F4 + F5 + F6 + F7 y guarda el
  mapa F5, el PDF y el historial SQLite.
- La demo muestra el score, la banda, las evidencias y el método de calibración.

## 8. Límites

1. **n = 50.** Los IC son anchos y la varianza entre folds puede ser alta. Hay
   un barrido de 5 semillas (PR-AUC 0.9800 ± 0.0010), pero F8 debe añadir
   robustez, McNemar y kappa.
2. **T7 trivial.** La copia exacta infla `f6_duplicate` y el salto de la ablación
   semántica. Falta T7 con reescaneo/recompresión.
3. **Riesgo de atajo sintético.** El generador deja pistas (metadatos,
   plantillas); el modelo puede aprenderlas. No hay validación con documentos
   reales del CETPRO.
4. **F5 aporta poco en T1–T4.** La fusión compensa con reglas y semántica, pero
   la localización fina sigue pendiente.
5. **Calibración identidad.** Es una decisión defendible con n pequeño, pero no
   equivale a una calibración institucional.
6. **Umbral max-F1 = 0.05.** Sube el recall a 0.96 sin cambiar el FPR en este
   corpus; es un artefacto del tamaño muestral, no un umbral recomendable sin
   validación externa.
