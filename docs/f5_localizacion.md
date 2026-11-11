# F5 — Localización y explicabilidad visual

> **Estado:** cerrado experimental. U-Net de página a 128 px, entrenada en CPU
> con el corpus sintético v2. Localiza bien T5/T6 (regiones grandes) y **falla
> T1–T4** (ediciones pequeñas de texto). Los números son una prueba de
> viabilidad sobre datos sintéticos, no un estimador institucional.

## 1. Qué se implementó

| Pieza | Archivo | Función |
|---|---|---|
| Modelo | `src/fraude/models/unet.py` | U-Net con encoder/decoder simple y `base_channels=8` |
| Dataset | `src/fraude/models/seg_dataset.py` | Página + máscara redimensionada con vecino más cercano |
| Entrenamiento | `src/fraude/models/seg_train.py` | 5 folds agrupados, `pos_weight`, oversampling de documentos con máscara |
| Evaluación | `src/fraude/models/seg_evaluate.py` | Pixel-AUC, Pixel-F1/IoU con umbral fijo, calibrado y oráculo; IC bootstrap |
| Inferencia | `src/fraude/models/seg_infer.py` | Máscara de probabilidad, resumen y superposición |
| Figuras | `src/fraude/models/seg_report.py` | Curvas, barrido de umbral, Pixel-F1 por ataque y ejemplos |

Comandos:

```bash
make train-f5 ARGS="--epochs 12 --device cpu --image-size 128"
make eval-f5
make figures-f5
```

## 2. Protocolo

- Validación cruzada agrupada por persona (5 folds), sin fuga entre documentos
  del mismo titular.
- El umbral calibrado de cada fold se elige con los otros folds; el oráculo usa
  el mismo OOF y solo sirve como techo diagnóstico.
- Se reporta Pixel-AUC global y media por imagen, Pixel-F1 e IoU con umbral fijo
  0.5, calibrado y oráculo, con IC bootstrap al 95 %.
- El corpus visual incluye T1–T6 (270 imágenes OOF; 120 con máscara positiva).
  T7/T8 no tienen máscara local y quedan fuera de F5.

## 3. Resultados (OOF)

| Métrica | Valor |
|---|---:|
| Imágenes OOF | 270 |
| Imágenes con máscara positiva | 120 |
| Píxeles positivos | 308 858 (6.98 %) |
| Pixel-AUC global | 0.9933 |
| Pixel-AUC media por imagen | 0.8266 ± 0.1666 |
| Pixel-F1 @0.5 | 0.8456 (IC 0.7474–0.8945) |
| Pixel-F1 calibrado | 0.9152 (IC 0.8701–0.9423) |
| IoU calibrado | 0.8436 (IC 0.7701–0.8909) |
| Pixel-F1 oráculo | 0.9505 |

### Por ataque (documentos base)

| Ataque | n | Pixel-F1 @0.5 | Pixel-F1 calibrado |
|---|---:|---:|---:|
| T1 sustitución de texto | 24 | 0.023 | 0.023 |
| T2 sello insertado | 24 | 0.017 | 0.016 |
| T3 copy-move | 18 | 0.123 | 0.002 |
| T4 borrado de campo | 18 | 0.037 | 0.031 |
| T5 reemplazo de foto | 18 | 0.563 | 0.799 |
| T6 documento fabricado | 18 | 0.992 | 0.971 |

### Por fold

| Fold | Umbral calibrado | n | Pixel-F1 | IoU |
|---|---:|---:|---:|---:|
| 0 | 0.60 | 54 | 0.0156 | 0.0079 |
| 1 | 0.80 | 54 | 0.9863 | 0.9729 |
| 2 | 0.80 | 54 | 0.9676 | 0.9372 |
| 3 | 0.80 | 60 | 0.9249 | 0.8602 |
| 4 | 0.80 | 48 | 0.0000 | 0.0000 |

Figuras: `artifacts/models/f5/figures/` (curvas, barrido, Pixel-F1 por ataque y
ejemplos cualitativos). El detalle completo está en
`artifacts/models/f5/eval.md`.

## 4. Lectura honesta

1. **El promedio alto engaña.** 0.915 de Pixel-F1 calibrado está dominado por
   T6 (página completa) y T5 (foto grande). En T1–T4 la localización es
   prácticamente nula.
2. **La causa es la resolución.** A 128 px, una línea de texto o un sello ocupan
   pocos píxeles; la máscara se pierde en el redimensionado. La solución
   correcta es entrenar por parches a mayor resolución, no subir el umbral.
3. **Los folds 0 y 4 no contienen T5/T6.** Su Pixel-F1 es cero porque no hay
   región grande que localizar; no es un fallo aleatorio del modelo.
4. **Hay falsos positivos cualitativos.** La figura de ejemplos muestra un
   auténtico rotado con una región marcada. F5 no debe usarse como decisión
   aislada; en la demo se presenta como ayuda cualitativa.
5. **T7/T8 no se evalúan aquí.** No dejan huella local; los cubren las reglas,
   el registro y F6.

## 5. Uso en la demo

La pestaña **Análisis integral** muestra el mapa F5 superpuesto y el resumen
`max_prob`, `mean_prob` y `area_fraction`. La fusión F7 solo recibe estos
resúmenes, no la máscara completa. El PDF incluye el mapa con la advertencia de
que es experimental.

## 6. Trabajo pendiente

- Entrenamiento por parches a 256/384 px con muestreo positivo de regiones.
- Folds estratificados por familia de ataque.
- Comparación CNN vs ViT vs VAE un-clase (E5-3).
- Integración de TruFor/ADCD-Net como señal preentrenada (E5-2).
