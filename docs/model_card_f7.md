# Model card — F7 (fusión calibrada)

## 1. Identificación

| Campo | Valor |
|---|---|
| Nombre | `fraude-fusion-f7` |
| Versión | 0.1.0 (prototipo de curso) |
| Tipo | Clasificador binario tabular (LightGBM) + calibrador de score |
| Artefacto | `artifacts/models/f7/fusion.joblib` |
| Entrada | Vector de 62 señales de F3–F6 (reglas, forense, F4, F5, semántica) |
| Salida | Score en [0, 1], banda de riesgo y evidencias SHAP |
| Fecha de la corrida | 2026-11-10 |
| Semilla | 42 |

## 2. Uso previsto

- **Sí:** priorizar revisión humana de documentos académicos sintéticos del
  CETPRO; generar evidencias explicables y un informe PDF/JSON.
- **Sí:** demostración docente e investigación aplicada.
- **No:** rechazar documentos automáticamente, certificar autenticidad,
  sustituir la revisión de RENIEC/MINEDU o tomar decisiones disciplinarias.
- **No:** usarse como probabilidad institucional de fraude. El score es una
  señal experimental sobre datos sintéticos.

## 3. Datos de entrenamiento

| Concepto | Valor |
|---|---|
| Corpus | `data/synthetic/v2` (50 documentos base: 25 auténticos / 25 manipulados) |
| Personas ficticias | 10 |
| Tipos documentales | 7 |
| Imágenes con aumentación | 300 |
| Ataques base | T1–T8 |
| Partición | 5 folds agrupados por persona; OOF para métricas |
| Extracción | OCR real docTR sobre las imágenes base |
| F5/F4 | Scores OOF para las filas de entrenamiento |

No se usaron documentos reales del CETPRO. No hay PII real.

## 4. Métricas (OOF, 50 documentos)

| Métrica | Valor |
|---|---:|
| PR-AUC | 0.9796 |
| ROC-AUC | 0.9760 |
| F1 @0.5 | 0.8980 |
| Precision @0.5 | 0.9167 |
| Recall @0.5 | 0.8800 |
| FPR @0.5 | 0.0800 |
| Brier (identidad) | 0.0854 |
| ECE (identidad) | 0.0700 |
| Matriz @0.5 | TP 22 · TN 23 · FP 2 · FN 3 |

La calibración elegida es **identidad** porque isotónica (ECE 0.102) y Platt
(ECE 0.138) empeoran con n = 50. Ver `docs/f7_fusion.md`.

## 5. Factores y sesgos conocidos

- `f6_duplicate` domina la importancia porque T7 del corpus es una copia
  exacta. El modelo puede sobrevalorar duplicados triviales.
- `registry_score` aparece con contribución positiva en algunos casos porque
  las manipulaciones derivan de documentos auténticos y conservan la identidad;
  el mensaje de la demo se redacta de forma neutral.
- El corpus tiene 10 personas y 7 tipos; la composición por ataque no está
  estratificada en los folds. Los folds 0 y 4 no contienen T5/T6.
- El generador sintético puede dejar atajos (metadatos, plantillas, fuentes).

## 6. Limitaciones

1. n = 50; semilla principal 42 y barrido de 5 semillas; los intervalos son anchos y falta robustez.
2. Sin documentos reales ni validación externa.
3. T7 no mide reutilización real (copia exacta).
4. F5 aporta poco en T1–T4.
5. El umbral max-F1 (0.05) es un artefacto del corpus; no se recomienda
   desplegarlo sin recalibrar.
6. El LLM no participa en la decisión; la explicación por plantilla es el
   fallback probado.

## 7. Consideraciones éticas y legales

- Ley N.° 29733: datos locales, sin envío a APIs externas.
- El sistema no rechaza documentos: solo prioriza revisión.
- Toda evidencia mostrada es trazable a una señal del pipeline.
- Los datos y artefactos están fuera de git (`data/`, `artifacts/`).

## 8. Reproducibilidad

```bash
make synth-v2
make validate-v2
make folds-v2
make pipeline-v2
make eval-f6
make features-f7
make train-f7
make figures-f7
```

Las métricas de este documento salen de `artifacts/models/f7/metrics.json` y
`artifacts/models/f7/eval.md`.
