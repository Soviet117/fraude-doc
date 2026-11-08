# Guía breve para presentar el prototipo

Este documento es apoyo para la sustentación del curso. No sustituye el plan,
el Cap. IV ni la model card; ordena qué mostrar, qué cifras se pueden citar y
qué límites hay que decir en voz alta.

## 1. Mensaje en una frase

El prototipo recibe un documento académico sintético, combina calidad, forense,
OCR, reglas, registro y un clasificador visual experimental, y devuelve
**evidencias para priorizar revisión humana**, nunca una decisión automática.

## 2. Estado de las fases

| Fase | Estado | Qué mostrar |
|---|---|---|
| F0 | Cerrada en local | Docker, CI, contrato `Observation` |
| F1 | Cerrada con corpus sintético | 20 muestras v1 + 50 documentos base v2; sin documentos reales |
| F2 | Cerrada | Calidad, generador, ataques T1–T8, aumentación, particiones |
| F3 | Cerrada provisional | Forense + OCR + reglas + registro; detecta T7/T8 |
| F4 | Cerrada experimental | EfficientNet-B0 de página; detecta T5/T6; falla T1–T4 |
| F5 | Pendiente | Localización/parches para T1–T4 |
| F6 | Pendiente | Rama semántica/NLP y explicación con LLM local |
| F7 | Pendiente | Fusión calibrada LightGBM |
| F8 | Pendiente | Evaluación integral con varias semillas y robustez |
| F9 | En curso | Demo, model card, data card y Cap. IV |

## 3. Demo en cinco minutos

### Antes de empezar

```bash
make test          # debe quedar en verde
make demo          # Streamlit en http://localhost:8501
```

Si no existe el corpus v2, generarlo una vez (tarda unos minutos):

```bash
make synth-v2
make validate-v2
make folds-v2
```

El checkpoint que usa la demo es `artifacts/models/f4/fold_2.pt`; si no existe:

```bash
make train-f4 ARGS="--epochs 12 --device cpu"
make eval-f4
```

### Guion

1. **Abrir la demo** y mostrar la barra lateral: origen del documento, muestra
   del corpus o archivo subido.
2. **F3 con M19 (T7):** seleccionar `M19 · DNI` en el menú. En la pestaña
   *Análisis* mostrar calidad, forense, OCR, reglas y el cruce con registro. La
   señal fuerte es la sustitución de identidad, no una huella visual.
3. **F3 con M20 (T8):** mostrar la inconsistencia de periodo/fecha. Explicar que
   F3 cubre semántica y registro; F4 no está pensado para eso.
4. **F4 con T5:** en la barra lateral elegir *Subir archivo* y subir
   `data/synthetic/v2/images/dni_0007_t5.png`. Ir a la pestaña *F4 experimental*:
   score alto y Grad-CAM. Es un cambio de fotografía, región grande.
5. **F4 con T6:** subir `data/synthetic/v2/images/certificado_modulo_0005_t6.png`.
   Mostrar score alto. T6 cambia la página completa.
6. **F4 con T1 (fallo honesto):** subir
   `data/synthetic/v2/images/dni_0000_t1.png`. Score bajo. Explicar que T1 es una
   edición local de texto y que un clasificador de página a 224 px no la ve; para
   eso está F5 (localización/parches).
7. **Pestaña *Fases*:** cerrar con el estado real: F0–F4 cerradas, F5–F9
   pendientes.

> Las muestras del menú lateral son del corpus v1 (`data/muestras`); el
> checkpoint F4 se entrenó con el corpus v2. Para la demo F4 conviene subir una
> imagen de `data/synthetic/v2/images/`, como se indica arriba.

### Figuras para diapositivas

```bash
make figures-f4    # curvas, matriz de confusión, recall por ataque y calibración
make gradcam-f4    # TP, TN, FP y FN
```

`figures-f4` deja cuatro PNGs en `artifacts/models/f4/figures/`. `gradcam-f4`
genera los mapas de calor en `artifacts/models/f4/gradcam/` usando el fold 2,
que es el de mejor PR-AUC de validación. Los casos son out-of-fold de ese fold.

## 4. Cifras que se pueden citar

### F4 — predicciones out-of-fold (270 imágenes, 45 documentos base)

| Métrica | Valor | IC 95 % |
|---|---:|---|
| PR-AUC | 0.712 | 0.641-0.778 |
| ROC-AUC | 0.653 | 0.581-0.721 |
| F1 @0.5 | 0.541 | 0.446-0.625 |
| Precision @0.5 | 0.803 | — |
| Recall @0.5 | 0.408 | — |
| FPR @0.5 | 0.080 | 0.040-0.124 |

La prevalencia (PR-AUC sin habilidad) es 0.444. El IC inferior del PR-AUC queda
por encima, así que hay señal; el ROC-AUC es modesto.

### F4 — recall por ataque (documentos base)

| Ataque | n | Recall @0.5 |
|---|---:|---:|
| T1 sustitución de texto | 4 | 0.000 |
| T2 sello insertado | 4 | 0.250 |
| T3 copy-move | 3 | 0.333 |
| T4 borrado de campo | 3 | 0.000 |
| T5 reemplazo de fotografía | 3 | 1.000 |
| T6 documento fabricado | 3 | 1.000 |

Los folds 0 y 4 quedan cerca del azar (PR-AUC 0.521 y 0.420) porque no contienen
T5/T6; los folds 1, 2 y 3 superan 0.77. La composición de ataques por fold
domina la varianza.

### F3 — línea base sin deep learning (`stripped`, 20 muestras)

| Métrica | Valor |
|---|---:|
| Precision | 0.500 |
| Recall | 0.125 |
| F1 | 0.200 |
| FPR | 0.083 |

F3 detecta T7/T8; F4 aporta T5/T6. La comparación **no es controlada**: distinto
corpus, tamaño y prevalencia. Sirve para mostrar complementariedad, no para
afirmar que F4 reemplaza a F3.

### Calibración F4

| Versión | Brier | ECE |
|---|---:|---:|
| Cruda | 0.2825 | 0.2649 |
| Isotónica cruzada | 0.2068 | 0.0858 |

La calibración mejora, pero la demo muestra el score crudo y lo declara como no
calibrado. No es una probabilidad de fraude.

## 5. Límites que hay que decir

1. **Todo es sintético.** No hay documentos reales del CETPRO; los números son
   prueba de viabilidad, no desempeño institucional.
2. **50 documentos base y 10 personas.** Los intervalos son anchos y la
   variabilidad entre folds es alta.
3. **Una sola semilla (42).** Repetir con otras semillas es F8.
4. **F4 es clasificación de página, no localización.** Grad-CAM es cualitativo;
   Pixel-F1/IoU son F5.
5. **T1–T4 no se detectan.** T1/T4 son ediciones locales de texto; T2/T3 quedan
   en 0.250/0.333. Es el argumento para F5.
6. **T7/T8 no están en F4.** Los cubren las reglas y el registro de F3, y F6
   para semántica.
7. **Ninguna decisión automática.** El sistema prioriza revisión humana; el LLM
   solo redactaría explicaciones y nunca decide.
8. **Comparación F3/F4 no controlada.** Distinto corpus y prevalencia.

## 6. Preguntas probables

**¿Por qué F1 no es más alto?**
Porque el umbral 0.5 es conservador (precision 0.803, recall 0.408, FPR 0.080).
El umbral calibrado max-F1 sube el recall a 0.983 pero el FPR a 0.987: marca casi
todo. El punto FPR ≤ 0.2 es un compromiso más razonable. No hay un umbral único
“correcto” sin coste de revisión.

**¿Por qué no entrenar con T7/T8?**
T7 reutiliza un documento auténtico de otra persona y T8 cambia contenido
semántico; no dejan huella visual local. Meterlos en una CNN de página añade
ruido de etiqueta. F3 y F6 son las ramas correctas.

**¿Por qué no usar un LLM para decidir?**
ADR-05: el LLM solo redacta a partir de hallazgos estructurados. Debe existir
un fallback por plantilla y el sistema no puede rechazar documentos solo.

**¿Por qué no hay documentos reales?**
Ley N.° 29733 y alcance del curso: no se usan datos personales reales. La
limitación está declarada en F1 y en la model card.

**¿Qué falta para producción?**
F5 (localización), F6 (semántica), F7 (fusión calibrada), F8 (evaluación
integral con semillas y robustez), F9 (cierre documental) y validación con el
CETPRO.

## 7. Guion de diapositivas (8)

1. **Problema y alcance:** fraude documental en CETPRO; apoyo a revisión, no
   decisión automática.
2. **Pipeline:** calidad → forense → OCR → extracción → reglas → registro →
   F4 → explicación.
3. **Corpus v2:** 50 documentos base, 25/25, 10 personas, 7 tipos, 300 imágenes,
   T1–T8.
4. **F3:** forense + OCR + reglas + registro; tabla `stripped`; detecta T7/T8.
5. **F4:** EfficientNet-B0, 224 px, folds agrupados, OOF, calibración.
6. **Resultados F4:** PR-AUC 0.712, F1 0.541, FPR 0.080; T5/T6 recall 1.0;
   T1/T4 0.
7. **Errores y límites:** folds 0/4, ataques locales, corpus sintético, una
   semilla, Grad-CAM no es localización.
8. **Roadmap:** F5–F9 y demo; complementariedad F3/F4/F6/F7.

## 8. Checklist antes de presentar

- [ ] `make test` en verde.
- [ ] `make lint` en verde.
- [ ] `make demo` abre en `localhost:8501`.
- [ ] `artifacts/models/f4/fold_2.pt` existe.
- [ ] `artifacts/models/f4/eval.md` y `eval.json` existen.
- [ ] `make figures-f4` y `make gradcam-f4` generaron las figuras.
- [ ] Las imágenes de demo son sintéticas y no contienen PII.
- [ ] Capturas de respaldo de la demo y de `eval.md` por si falla el directo.
- [ ] Cerrar otras aplicaciones que consuman CPU/GPU antes de la demo.
