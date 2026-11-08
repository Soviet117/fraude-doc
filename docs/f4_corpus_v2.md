# Corpus sintético v2 y F4 experimental

Este documento describe el corpus de 50 documentos sintéticos construido para
entrenar el clasificador visual de página (F4) y el protocolo mínimo con el que
se evalúa. Las secciones 1–2 hacen de **data card**; la model card resumida está
en [`docs/model_card_f4.md`](model_card_f4.md). **No sustituye** la validación
con documentos reales del CETPRO ni la localización de F5.

## 1. Composición

| Concepto | Cantidad |
|---|---:|
| Documentos base auténticos | 25 |
| Documentos base manipulados | 25 |
| Documentos base totales | 50 |
| Personas ficticias | 10 |
| Tipos documentales | 7 |
| Copias por documento (5 aumentaciones + original) | 6 |
| Imágenes totales | 300 |

Cada manipulado comparte `source_id` con su documento auténtico de origen. La
aumentación de captura también conserva `source_id` y `origin_id`, de modo que
ninguna copia del mismo documento puede caer en un fold distinto.

### Distribución por ataque (documentos base)

| Ataque | Documentos |
|---|---:|
| T1 sustitución de texto | 4 |
| T2 sello insertado | 4 |
| T3 copy-move | 3 |
| T4 borrado de campo | 3 |
| T5 reemplazo de fotografía | 3 |
| T6 documento fabricado | 3 |
| T7 reutilización de identidad | 3 |
| T8 inconsistencia semántica | 2 |
| **Total** | **25** |

T5 solo puede aplicarse a DNI; el corpus contiene 10 DNI base, suficientes para
las 3 copias. T7 reutiliza los píxeles de un documento de otra persona y se
etiqueta como manipulado porque la identidad no corresponde al titular.

### Distribución por tipo documental (documentos base)

| Tipo | Documentos |
|---|---:|
| DNI | 10 |
| Certificado de estudios | 8 |
| Partida de nacimiento | 8 |
| Ficha de matrícula | 8 |
| Acta de evaluación | 6 |
| Certificado de módulo | 5 |
| Título de Auxiliar Técnico | 5 |

## 2. Generación reproducible

```bash
make synth-v2       # 50 documentos base + 250 aumentados + manifiestos
make validate-v2    # integridad, hashes, balance y particiones
make folds-v2       # folds agrupados de 5 vías
```

La semilla es 42 y queda registrada en `build.json`. El comando no usa red ni
servicios externos: todo se genera localmente con Pillow y ReportLab.

Archivos producidos en `data/synthetic/v2/` (ignorado por git):

| Archivo | Contenido |
|---|---|
| `manifest.jsonl` | 25 auténticos con campos, cajas y hashes |
| `attacks_manifest.jsonl` | 25 manipulados con `parent_id`, región y máscara |
| `augmented_manifest.jsonl` | 250 copias de captura |
| `split_manifest.jsonl` | 300 registros con partición fija |
| `split_summary.json` | conteos por partición, ataque y tipo |
| `folds_manifest.jsonl` | fold 0–4 por muestra (validación cruzada) |
| `folds_summary.json` | tamaño y clases por fold |
| `build.json`, `dataset.json` | resumen de la construcción |

### Partición fija (compatibilidad con F2/F3)

| Partición | Imágenes | Personas |
|---|---:|---:|
| train | 102 | 4 |
| val | 36 | 1 |
| calibration | 36 | 1 |
| test | 36 | 1 |
| unseen_attack (T5) | 90 | 3 |

La partición fija queda muy cargada hacia `unseen_attack` porque T5 toca tres
personas y todas sus muestras viajan juntas para no filtrar identidad. Para F4
se usa validación cruzada agrupada de 5 vías sobre las 10 personas; la partición
fija se conserva para comparar con F3 y para un chequeo descriptivo de T5.

### Validación cruzada agrupada (subconjunto visual T1–T6)

| Fold | Imágenes | Auténticas | Manipuladas | Personas |
|---|---:|---:|---:|---:|
| 0 | 54 | 30 | 24 | 2 |
| 1 | 54 | 30 | 24 | 2 |
| 2 | 54 | 30 | 24 | 2 |
| 3 | 60 | 30 | 30 | 2 |
| 4 | 48 | 30 | 18 | 2 |

El reparto agrupa personas completas y no parte documentos, por eso los folds
no son exactamente iguales. Las métricas se reportan por fold (media ±
desviación) y agregadas sobre las predicciones out-of-fold.

## 3. Entrenamiento y evaluación

El clasificador visual se entrena con T1–T6. T7 y T8 quedan fuera del
entrenamiento visual porque no dejan una huella local: T7 reutiliza un
documento auténtico de otra persona y T8 cambia contenido semántico. Esas dos
familias ya las cubren las reglas y el cruce con registro de F3; forzar a una
CNN de página a aprenderlas introduciría ruido de etiqueta y métricas
engañosas. El conjunto visual queda en 25 auténticos + 20 manipulados base
(45 documentos, 270 imágenes con aumentación).

```bash
# Humo en CPU: 2 folds, 1 época, sin pesos preentrenados
uv run python -m fraude.models.train --data data/synthetic/v2 \
  --out artifacts/models/f4_smoke --folds 2 --epochs 1 --image-size 128 \
  --cache-size 160 --batch-size 8 --no-pretrained --device cpu --limit-groups 2

# Corrida completa (CPU o GPU)
make train-f4 ARGS="--epochs 12 --device cpu"
make eval-f4
make figures-f4
make gradcam-f4
```

El entrenamiento usa EfficientNet-B0 de `timm`, una salida logit, `BCEWithLogitsLoss`
con `pos_weight`, AdamW, cosine annealing, AMP solo si hay CUDA y early stopping
sobre PR-AUC de validación. Las imágenes se cachean a 256 px con letterbox para
no deformar la página; el recorte de 224 px es aleatorio en train y central en
validación. No se usan flips horizontales porque invertirían el texto. Cada
checkpoint conserva una copia de los pesos del mejor epoch según PR-AUC de
validación; el OOF se recalcula con esos pesos, no con la última época.

La demo usa `fold_2.pt`, el fold con mejor PR-AUC de validación, solo para la
inspección cualitativa. Las métricas de la sección 6 no salen de ese checkpoint:
salen de las predicciones out-of-fold de los cinco folds.

Salidas en `artifacts/models/f4/`:

| Archivo | Contenido |
|---|---|
| `fold_*.pt` | pesos del mejor epoch por fold |
| `oof.jsonl` | predicción out-of-fold por muestra |
| `metrics.json` | configuración, historial, métricas por fold y agregadas |
| `eval.md` | tabla legible de umbrales, calibración, ataques y folds |
| `figures/` | curvas de entrenamiento, matriz de confusión, recall por ataque y calibración |
| `gradcam/` | mapas de calor de TP, TN, FP y FN |

## 4. Protocolo de evaluación

- Umbral fijo 0.5, umbral calibrado @0.5, umbral calibrado que maximiza F1,
  umbral calibrado con FPR ≤ 0.2 y umbral oráculo (plan §7.4).
- El umbral oráculo se calcula sobre el mismo OOF y solo sirve como techo
  diagnóstico; no es un umbral desplegable.
- Calibración isotónica ajustada de forma cruzada: cada fold se calibra con los
  otros folds, nunca con sus propias predicciones.
- Brier y ECE crudos y calibrados.
- IC bootstrap al 95 % para F1, FPR, PR-AUC y ROC-AUC.
- Desglose por ataque y por tipo documental; un promedio alto puede ocultar un
  ataque que falla siempre.
- Media ± desviación entre folds.

## 5. Límites declarados

1. **50 documentos no estiman generalización.** Los intervalos de confianza
   serán anchos y la variabilidad entre folds es parte del resultado.
2. **Todo es sintético.** Las plantillas, firmas y fotos las genera el código;
   el modelo puede aprender atajos del generador. Cualquier número debe leerse
   como prueba de viabilidad, no como desempeño sobre documentos del CETPRO.
3. **Clasificación de página, no localización.** Grad-CAM es una ayuda
   cualitativa; Pixel-F1 e IoU son F5 y no se reportan aquí.
4. **T7 y T8 quedan fuera del clasificador visual.** T7 reutiliza un documento
   auténtico de otra persona y T8 cambia contenido semántico sin dejar huella
   local evidente. Se reportan como límite de alcance, no como fallo de F4: la
   detección de esas familias corresponde a F3 (reglas y registro) y a F6
   (rama semántica).
5. **Sin documentos reales.** La limitación de F1 sigue vigente: no hay
   validación con el CETPRO ni anotación de expertos.
6. **Una sola semilla.** La corrida usa semilla 42; repetir con otras semillas
   queda como trabajo inmediato antes de citar media ± desviación entre
   semillas (el plan §11.2 lo exige para la evaluación integral, F8).
7. **Folds no estratificados por ataque.** Con 10 personas, los folds 0 y 4 no
   contienen T5/T6, las dos familias con huella visual de página. Eso explica
   su PR-AUC cercana al azar y no se corrigió después de ver los resultados:
   estratificar los folds por familia queda como trabajo futuro y debe
   decidirse antes de la siguiente corrida.

## 6. Resultados

Corrida del 2026-11-07: 5 folds, hasta 12 épocas por fold, EfficientNet-B0 a
224 px, CPU y semilla 42. Se evaluaron 270 predicciones out-of-fold (45
documentos base). La tabla cruda está en `artifacts/models/f4/eval.md` y el JSON
completo en `artifacts/models/f4/eval.json`. Se reproduce con:

```bash
make train-f4 ARGS="--epochs 12 --device cpu"
make eval-f4
```

### 6.1 Agregado

| Métrica | Valor | IC 95 % |
|---|---:|---|
| PR-AUC | 0.712 | 0.641-0.778 |
| ROC-AUC | 0.653 | 0.581-0.721 |
| F1 @0.5 | 0.541 | 0.446-0.625 |
| Precision @0.5 | 0.803 | — |
| Recall @0.5 | 0.408 | — |
| FPR @0.5 | 0.080 | 0.040-0.124 |
| Prevalencia (PR-AUC sin habilidad) | 0.444 | — |

El PR-AUC supera la prevalencia y su IC inferior (0.641) queda por encima de
0.444: hay señal visual, pero el ROC-AUC de 0.653 es modesto. Con el umbral
fijo 0.5 el modelo es conservador (precision 0.803, recall 0.408, FPR 0.080):
marca pocos auténticos como manipulados, a costa de dejar pasar muchas
manipulaciones.

### 6.2 Umbrales

| Escenario | Umbral | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Fijo 0.5 | 0.50 | 0.803 | 0.408 | 0.541 | 0.080 |
| Calibrado @0.5 | 0.50 | 0.815 | 0.367 | 0.506 | 0.067 |
| Calibrado max-F1 | 0.25 | 0.444 | 0.983 | 0.611 | 0.987 |
| Calibrado FPR≤0.2 | 0.39 | 0.671 | 0.442 | 0.533 | 0.173 |
| Oráculo | 0.05 | 0.615 | 0.533 | 0.571 | 0.267 |

El umbral calibrado max-F1 (0.25) sube el recall a 0.983 pero también el FPR a
0.987: en la práctica marca casi todo como manipulado y no sirve para priorizar
revisión. El umbral calibrado con FPR ≤ 0.2 (0.39) es un punto de operación más
razonable (F1 0.533, FPR 0.173). El oráculo (0.05) solo alcanza F1 0.571: el
techo es bajo y no se arregla moviendo el umbral.

### 6.3 Calibración

| Versión | Brier | ECE |
|---|---:|---:|
| Cruda | 0.2825 | 0.2649 |
| Isotónica cruzada | 0.2068 | 0.0858 |

La isotónica cruzada reduce el Brier de 0.2825 a 0.2068 y el ECE de 0.2649 a
0.0858. Mejora la calibración, pero no convierte el score en una probabilidad
institucional: el calibrador se ajusta con los otros folds del mismo corpus
sintético. La demo muestra el score crudo y lo declara como no calibrado.

### 6.4 Por ataque (documentos base)

| Grupo | n | Recall @0.5 | FPR @0.5 |
|---|---:|---:|---:|
| T1 sustitución de texto | 4 | 0.000 | — |
| T2 sello insertado | 4 | 0.250 | — |
| T3 copy-move | 3 | 0.333 | — |
| T4 borrado de campo | 3 | 0.000 | — |
| T5 reemplazo de fotografía | 3 | 1.000 | — |
| T6 documento fabricado | 3 | 1.000 | — |
| Auténticos | 25 | — | 0.080 |

El patrón es el esperado para un clasificador de página: detecta T5 y T6, que
cambian regiones grandes o la estructura completa, y falla en T1 y T4, ediciones
locales de texto invisibles a 224 px. T2 y T3 quedan en 0.250 y 0.333. Esto no
es un bug que ocultar: es la evidencia de que F4 necesita la localización y el
entrenamiento por parches de F5 para las manipulaciones pequeñas.

### 6.5 Por fold

| Fold | n | PR-AUC | F1 @0.5 | Ataques en validación |
|---|---:|---:|---:|---|
| 0 | 54 | 0.521 | 0.148 | T1, T2, T3, T4 |
| 1 | 54 | 0.772 | 0.615 | T1, T4, T5, T6 |
| 2 | 54 | 0.844 | 0.762 | T1, T2, T5, T6 |
| 3 | 60 | 0.819 | 0.691 | T1, T2, T3, T5, T6 |
| 4 | 48 | 0.420 | 0.000 | T2, T3, T4 |

Media entre folds: PR-AUC 0.675 ± 0.192; F1 0.443 ± 0.345. Los folds 0 y 4 son
casi aleatorios y son precisamente los que no contienen T5/T6. Los folds 1, 2 y
3, que sí los contienen, superan 0.77 de PR-AUC. La varianza no es solo del
tamaño: la composición de ataques por fold domina el resultado.

### 6.6 Por tipo documental (con aumentación)

| Tipo | n | PR-AUC | Recall @0.5 | FPR @0.5 |
|---|---:|---:|---:|---:|
| DNI | 48 | 0.937 | 0.800 | 0.111 |
| Título de Auxiliar Técnico | 24 | 0.976 | 1.000 | 0.056 |
| Certificado de módulo | 30 | 0.807 | 0.500 | 0.000 |
| Partida de nacimiento | 48 | 0.707 | 0.389 | 0.067 |
| Certificado de estudios | 36 | 0.521 | 0.333 | 0.083 |
| Acta de evaluación | 36 | 0.504 | 0.000 | 0.111 |
| Ficha de matrícula | 48 | 0.463 | 0.083 | 0.125 |

La diferencia sigue la mezcla de ataques de cada tipo (el DNI concentra T5),
no una dificultad intrínseca del formato. No debe citarse como “el modelo
detecta DNI” sin decir que el corpus sintético asigna T5 solo a DNI.

### 6.7 Lectura para la presentación

- **Qué se puede afirmar:** F4 aprendió señal visual en datos sintéticos,
  especialmente en T5/T6, y la calibración mejora Brier/ECE. El pipeline
  reproducible existe y las métricas salen de comandos del repo.
- **Qué no se puede afirmar:** que detecte T1–T4, que generalice a documentos
  reales del CETPRO, que sea un localizador (F5) o que su score sea una
  probabilidad de fraude. T7/T8 no se evalúan en esta rama visual.
- **Siguiente paso técnico:** F5 con parches/localización para T1–T4 y una
  partición estratificada por familia de ataque; luego F6/F7 para semántica y
  fusión calibrada. La demo debe seguir presentándose como apoyo a la revisión
  humana, nunca como decisión automática.

### 6.8 Comparación con F3 (no controlada)

| Rama | Corpus | n | Precision | Recall | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|
| F3 `stripped` | `data/muestras` | 20 | 0.500 | 0.125 | 0.200 | 0.083 |
| F4 OOF @0.5 | corpus v2 visual | 270 | 0.803 | 0.408 | 0.541 | 0.080 |

F4 supera el piso heurístico de F3 en el corpus sintético, pero **no es una
comparación controlada**: distinto corpus, distinto tamaño y distinta
prevalencia. F3 sigue siendo la única rama que detecta T7/T8 en la evaluación
end-to-end; F4 aporta T5/T6. La comparación honesta es de complementariedad,
no de “F4 reemplaza a F3”. Una comparación pareada exigiría el mismo conjunto
y las mismas particiones, y queda para F8.
