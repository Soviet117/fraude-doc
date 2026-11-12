# Guía breve para presentar el prototipo

Apoyo para la sustentación. No sustituye el Cap. IV ni las model cards. Ordena
qué mostrar, qué cifras se pueden citar y qué límites hay que decir en voz alta.

## 1. Mensaje en una frase

El prototipo recibe un documento académico sintético, combina calidad, forense,
OCR, reglas, registro, semántica y modelos visuales, y devuelve **evidencias
para priorizar revisión humana**, nunca una decisión automática.

## 2. Estado de las fases

| Fase | Estado | Qué mostrar |
|---|---|---|
| F0 | Cerrada en local | Docker, CI, contrato `Observation` |
| F1 | Cerrada con corpus sintético | v1 (20) + v2 (50 documentos base); sin reales |
| F2 | Cerrada | Calidad, generador, T1–T8, aumentación, folds |
| F3 | Cerrada provisional | Forense + OCR + reglas + registro; T7/T8 |
| F4 | Cerrada experimental | EfficientNet-B0 de página; T5/T6 |
| F5 | Cerrada experimental | U-Net 128 px; T5/T6, limitado en T1–T4 |
| F6 | Cerrada experimental | Semántica, duplicados, explicación con fallback |
| F7 | Cerrada experimental | LightGBM + calibración; F1 0.898 OOF |
| F8 | Parcial | OOF, IC, ablaciones y 5 semillas; faltan robustez y kappa |
| F9 | En curso | Demo integral, SQLite, PDF, diagramas y documentación |

La demo es la fuente de verdad del estado; **Ayuda → Estado** lo muestra.

## 3. Demo en 6 minutos

### Antes de empezar

```bash
make test                      # debe quedar en verde
make demo                      # http://localhost:8501
```

Si faltan artefactos:

```bash
make synth-v2 && make validate-v2 && make folds-v2
make train-f4 ARGS="--epochs 12 --device cpu" && make eval-f4
make train-f5 ARGS="--epochs 12 --device cpu --image-size 128" && make eval-f5
make pipeline-v2 && make eval-f6 && make features-f7 && make train-f7
```

### Guion

1. **Abrir la demo** y mostrar la barra lateral: navegación, las tres opciones
   de origen y el modo oscuro. Explicar que el v2 entrena F4/F5/F7 y el v1
   conserva un ataque de cada tipo.
2. **Auténtico (`dni_0000`)**: vista *Análisis*. Mostrar banda baja, score,
   evidencias y mapa F5. Aclarar que el mapa es cualitativo y que los iconos
   `?` explican los términos.
3. **T7 (`dni_0007_t7_c1`)**: evidencia de reutilización y score alto. Decir que
   T7 es una copia exacta en el corpus; no mide reutilización real.
4. **T1 (`dni_0000_t1`)**: F4/F5 no ven la edición pequeña, pero la regla de
   registro sí; la fusión sube el riesgo. Es el argumento del sistema híbrido.
5. **T6 (`certificado_modulo_0005_t6`)**: score alto y mapa F5 cubriendo la
   página. Mostrar que el tipo detectado puede fallar y que la fusión compensa.
6. **Descargar PDF/JSON** y mostrar evidencias, campos, explicación y límites.
7. **Historial**: análisis guardado en SQLite; registrar una revisión humana.
8. **Ayuda → Estado**: cerrar con F0–F7 cerradas, F8 parcial y F9 en curso.

> Para probar T5/T6, elegir `dni_0007_t5` o `certificado_modulo_0005_t6` en el
> corpus v2; para T1, `dni_0000_t1`.

### Figuras para diapositivas

```bash
make figures-f4     # curvas, matriz, recall por ataque, calibración
make figures-f5     # Pixel-F1 por ataque, barrido, ejemplos
make figures-f7     # ROC/PR, confusión, calibración, ablaciones, SHAP
make seeds-f7       # barrido de semillas de F7
make eval           # reproduce las tablas y figuras OOF
make eval-system    # sanity check del pipeline integral
make diagrams       # casos de uso, arquitectura, ER y flujo
```

Salidas en `artifacts/models/f4/figures`, `artifacts/models/f5/figures`,
`artifacts/models/f7/figures` y `docs/figuras`.

Diagramas listos para diapositivas: `casos_uso.png`, `casos_uso_tabla.png`,
`modulos.png`, `clases.png`, `secuencia_analisis.png`, `estados_riesgo.png`,
`despliegue.png`, `base_datos_fisica.png`, `base_datos_datos.png`,
`er_historial.png`, `flujo_pipeline.png` y `casos_prueba.png`.

## 4. Cifras que se pueden citar

### F3 — línea base (`stripped`, 20 muestras)

| Precision | Recall | F1 | FPR |
|---:|---:|---:|---:|
| 0.500 | 0.125 | 0.200 | 0.083 |

### F4 — página (270 OOF)

| Métrica | Valor | IC 95 % |
|---|---:|---|
| PR-AUC | 0.712 | 0.641–0.778 |
| F1 @0.5 | 0.541 | 0.446–0.625 |
| Precision | 0.803 | — |
| Recall | 0.408 | — |
| FPR | 0.080 | 0.040–0.124 |

Recall T5 = 1.000, T6 = 1.000; T1/T4 = 0.000.

### F5 — localización (270 OOF)

| Escenario | Pixel-F1 | IoU |
|---|---:|---:|
| Fijo 0.5 | 0.846 | 0.733 |
| Calibrado | 0.915 | 0.844 |
| Oráculo | 0.951 | 0.906 |

T1–T4 ≈ 0; T5 calibrado 0.799; T6 0.971. **El promedio está dominado por T5/T6.**

### F6 — semántica (50 documentos)

| Métrica | Valor |
|---|---:|
| Tipo documental | 0.960 |
| Recall T7 | 1.000 (copia exacta) |
| FPR auténticos (T7) | 0.000 |
| Explicaciones no vacías | 50/50 |

### F7 — fusión (50 OOF)

| Métrica | Valor | IC 95 % |
|---|---:|---|
| PR-AUC | 0.980 | 0.946–1.000 |
| F1 @0.5 | 0.898 | 0.808–0.972 |
| Precision | 0.917 | — |
| Recall | 0.880 | — |
| FPR | 0.080 | — |
| Brier / ECE | 0.085 / 0.070 | — |

Calibración elegida: **identidad**, porque isotónica (ECE 0.102) y Platt
(ECE 0.138) empeoran con n = 50.

### Sistema

Latencia total ≈ **3.4 s** en CPU (OCR ≈ 2–3 s, forense ≈ 0.4 s, F4+F5 ≈ 0.1 s,
F7 ≈ 0.003 s).

## 5. Límites que hay que decir

1. **Todo es sintético.** No hay documentos reales del CETPRO; los números son
   prueba de viabilidad, no desempeño institucional.
2. **50 documentos base y 10 personas.** IC anchos; semilla principal 42 y barrido de 5 semillas, pero faltan robustez y kappa.
3. **F5 no localiza T1–T4.** Ediciones pequeñas de texto se pierden a 128 px.
4. **T7 es trivial.** La copia exacta infla la ablación semántica; falta
   reescaneo/recompresión.
5. **F4 no ve T1–T4.** Es un clasificador de página, no un localizador.
6. **Calibración identidad.** Es una decisión defendible con n pequeño, no una
   probabilidad institucional.
7. **F8 incompleta.** Faltan varias semillas, robustez, McNemar, kappa y pre/post.
8. **Comparación F3/F4/F5/F7 no controlada.** Distinto corpus, tamaño y
   prevalencia.
9. **Sin despliegue, sin RENIEC/MINEDU, sin blockchain.** Trabajo futuro.

## 6. Preguntas probables

**¿Por qué F7 da 0.98 si F5 falla T1–T4?**
Porque F7 no usa solo F5: combina reglas, forense, registro, semántica y F4/F5.
En T1, por ejemplo, la regla de registro detecta el cambio de DNI aunque la
localización no vea la edición. El 0.98 es sobre 50 documentos sintéticos; no
generaliza sin validación externa.

**¿Por qué no entrenar F5 a mayor resolución?**
Es el siguiente paso, pero en CPU y con 2 días de plazo se priorizó el sistema
funcional. Entrenar por parches a 256/384 px es trabajo de F5 v2/F8.

**¿Por qué la calibración es identidad?**
Con 50 casos, isotónica y Platt sobreajustan y suben Brier/ECE. El protocolo
compara los tres métodos y elige el de menor Brier; identidad gana (ECE 0.070).

**¿Por qué no usar un LLM para decidir?**
ADR-05: el LLM solo redacta a partir de hallazgos estructurados. La plantilla
funciona con Ollama apagado y la decisión es humana.

**¿Qué falta para producción?**
F8 completa (semillas, robustez, kappa), F5 por parches, documentos reales
anonimizados, validación con el CETPRO, verificación en las 3 PCs y revisión
legal.

## 7. Guion de diapositivas (10)

1. Problema y alcance: fraude documental en CETPRO; apoyo a revisión.
2. Pipeline híbrido: calidad → forense → OCR/reglas → semántica → F4/F5 → F7.
3. Corpus v2: 50 documentos, 25/25, 10 personas, 7 tipos, 300 imágenes.
4. F3: línea base; detecta T7/T8.
5. F4: EfficientNet-B0 de página; T5/T6, falla T1–T4.
6. F5: U-Net; Pixel-F1 alto por T5/T6, nulo en T1–T4 (límite honesto).
7. F6: semántica, duplicados y explicación con fallback.
8. F7: fusión calibrada; F1 0.898, ECE 0.070, ablaciones.
9. Demo: integral, PDF, SQLite, historial y revisión humana.
10. Límites y trabajo futuro: sintético, n pequeño, F8, F5 por parches.

## 8. Checklist antes de presentar

- [ ] `make test` en verde.
- [ ] `make lint` en verde.
- [ ] `make demo` abre en `localhost:8501`.
- [ ] Existen `artifacts/models/f4/fold_2.pt`, `artifacts/models/f5/fold_0.pt` y
      `artifacts/models/f7/fusion.joblib`.
- [ ] `artifacts/models/f7/metrics.json` y `eval.md` existen.
- [ ] `make eval` reprodujo las tablas OOF y `make figures-f4`,
      `make figures-f5`, `make figures-f7` y `make diagrams` generaron las figuras.
- [ ] Las imágenes de demo son sintéticas y no contienen PII.
- [ ] Capturas de respaldo de la demo y de `eval.md` por si falla el directo.
- [ ] Cerrar otras aplicaciones que consuman CPU/GPU antes de la demo.
