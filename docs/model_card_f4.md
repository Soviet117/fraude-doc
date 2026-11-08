# Model card — F4 clasificador visual de página

**Versión:** 0.1 experimental  
**Fecha:** 2026-11-07  
**Rama:** F4 del prototipo de detección de fraude documental del CETPRO San Agustín  
**Estado:** prueba de viabilidad con datos sintéticos; no es un modelo de producción

## 1. Qué hace

F4 recibe una imagen de página completa de un documento académico sintético y
devuelve un score entre 0 y 1 para la clase “manipulado”. Se usa como una señal
visual experimental dentro de la demo; no decide, no rechaza documentos y no
valida identidad.

## 2. Detalles del modelo

| Elemento | Valor |
|---|---|
| Arquitectura | EfficientNet-B0 (`timm`), una salida logit |
| Parámetros entrenables | 4 008 829 |
| Entrada | página con letterbox a 256 px y recorte de 224 px |
| Pesos iniciales | ImageNet (`efficientnet_b0.ra_in1k`) |
| Pérdida | `BCEWithLogitsLoss` con `pos_weight` |
| Optimizador | AdamW, lr 3e-4, weight decay 1e-4 |
| Scheduler | cosine annealing |
| Batch | 16 |
| Épocas máximas | 12 |
| Early stopping | paciencia 4 sobre PR-AUC de validación |
| Semilla | 42 |
| Dispositivo de la corrida | CPU |
| Checkpoint de demo | `fold_2.pt` (mejor PR-AUC de validación) |

Cada fold guarda los pesos del mejor epoch según PR-AUC de validación; el OOF se
recalcula con esos pesos. No se usan flips horizontales porque invertirían el
texto.

## 3. Datos de entrenamiento

- Corpus sintético v2: 50 documentos base (25 auténticos y 25 manipulados), 10
  personas ficticias, 7 tipos documentales y 300 imágenes con aumentación de
  captura.
- F4 usa solo T1–T6: 25 auténticos + 20 manipulados base (45 documentos, 270
  imágenes).
- T7/T8 se excluyen porque no dejan huella visual local; los cubren F3 y F6.
- Validación cruzada agrupada de 5 vías por `source_id`; ninguna persona cae en
  dos folds.
- Composición, generación y límites: [`docs/f4_corpus_v2.md`](f4_corpus_v2.md).

## 4. Evaluación

| Métrica | Valor | IC 95 % |
|---|---:|---|
| PR-AUC | 0.712 | 0.641-0.778 |
| ROC-AUC | 0.653 | 0.581-0.721 |
| F1 @0.5 | 0.541 | 0.446-0.625 |
| Precision @0.5 | 0.803 | — |
| Recall @0.5 | 0.408 | — |
| FPR @0.5 | 0.080 | 0.040-0.124 |

- 270 predicciones out-of-fold; prevalencia 0.444.
- Recall por ataque base: T5 1.000, T6 1.000, T3 0.333, T2 0.250, T1 0.000,
  T4 0.000.
- Calibración isotónica cruzada: Brier 0.2825 → 0.2068; ECE 0.2649 → 0.0858.
- La demo muestra el score crudo, no una probabilidad calibrada.

## 5. Usos previstos

- Demostración académica de un pipeline de apoyo a revisión humana.
- Priorización de documentos para revisión manual en el prototipo.
- Investigación sobre señales visuales en documentos sintéticos.

## 6. Usos no previstos

- Rechazar o acusar automáticamente a una persona.
- Validar oficialmente DNI, partidas, certificados o títulos.
- Sustituir la revisión de RENIEC, MINEDU o el CETPRO.
- Documentos reales con PII fuera del entorno local.
- Localización de la manipulación: F4 clasifica la página, no la región; eso es
  F5.
- Detección de T7/T8: esas familias corresponden a F3/F6.

## 7. Limitaciones

1. **Datos sintéticos.** El modelo puede aprender atajos del generador; los
   números no estiman desempeño institucional.
2. **Muestra pequeña.** 50 documentos base y 10 personas; los IC son anchos.
3. **Una sola semilla.** No hay media entre semillas; eso es F8.
4. **T1–T4 fallan.** Son ediciones locales; un clasificador de página a 224 px no
   las ve. Es la motivación de F5.
5. **Folds no estratificados.** Los folds 0 y 4 no contienen T5/T6 y quedan cerca
   del azar.
6. **Cambio de dominio v1/v2.** La demo usa muestras v1 (`data/muestras`) y el
   checkpoint se entrenó con v2; para F4 conviene subir imágenes de
   `data/synthetic/v2/images/`.
7. **Grad-CAM no es localización.** Es una ayuda cualitativa; Pixel-F1/IoU son
   F5.

## 8. Consideraciones éticas y legales

- Solo datos sintéticos; no se usan documentos reales ni PII.
- No se envían imágenes a APIs externas.
- El sistema prioriza revisión humana y no toma decisiones automáticas.
- Ley N.° 29733: cualquier documento real futuro exige anonimización y acuerdo
  con el CETPRO.

## 9. Reproducción

```bash
make synth-v2
make validate-v2
make folds-v2
make train-f4 ARGS="--epochs 12 --device cpu"
make eval-f4
make figures-f4
make gradcam-f4
```

Salidas en `artifacts/models/f4/`: checkpoints, `oof.jsonl`, `metrics.json`,
`eval.md`, `eval.json`, `figures/` y `gradcam/`.
