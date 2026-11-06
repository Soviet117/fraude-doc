# Evaluación end-to-end del corpus de muestras

**Fase:** F3 (línea base sin deep learning)  
**Fecha de corrida:** 2026-11-05  
**Corpus:** `data/muestras` — 20 documentos generados por IA: 12 auténticos y 8 manipulados (T1–T8), 12 personas ficticias y 7 tipos documentales  
**Artefacto:** `artifacts/eval/samples_pipeline.json` (ignorado por git)  
**Semilla:** 42 · **Bootstrap:** 2000 remuestras · **Umbral fijo:** 0.50

Esta evaluación ejecuta el **pipeline completo** sobre cada PNG: calidad, forense
clásico, OCR real con docTR, extracción de campos, reglas y cruce con el registro
sintético. No usa campos oráculo. Es distinta de la tabla de
[`docs/f3_linea_base.md`](f3_linea_base.md), que aísla reglas y registro con campos
del manifiesto.

El corpus es sintético y no representa el desempeño institucional del CETPRO. Hay una
sola muestra por ataque, así que **no existe recall por ataque defendible**.

---

## 1. Protocolo

| Elemento | Valor |
|---|---|
| Entrada | PNG de 300 dpi en `data/muestras/png` |
| Etiquetas | `data/muestras/_indice.csv` (`authentic` / `tampered`, ataque T1–T8) |
| Registro | `data/muestras/_personas.csv` (12 personas ficticias) |
| Score | Suma de pesos F3 por severidad, truncada a 1.0 |
| Umbral | 0.50 fijo; no calibrado |
| Modo `as_is` | PNG original, conserva los metadatos del generador |
| Modo `stripped` | Copia temporal sin chunks de texto/EXIF |
| Errores | 0/20 en ambos modos |
| Tipo documental | 20/20 correcto en ambos modos |

El modo `stripped` sirve para separar el **atajo de metadatos** (Photoshop) de las
señales que sobreviven al contenido. No es una simulación de un documento real: es el
mismo PNG sin metadatos.

---

## 2. Resultados

### 2.1 Tabla principal

| Modo | TP | TN | FP | FN | Precision | Recall | F1 | FPR | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `as_is` | 2 | 11 | 1 | 6 | 0.667 | 0.250 | 0.364 | 0.083 | 0.875 | 0.927 |
| `stripped` | 1 | 11 | 1 | 7 | 0.500 | 0.125 | 0.200 | 0.083 | 0.446 | 0.578 |

El único positivo en `stripped` es M19 (T7, sustitución de identidad). En `as_is` se
suman M20 (T8, periodo de matrícula alterado) y el falso positivo M12 (título auténtico
con una fecha que el OCR leyó mal).

### 2.2 Intervalos bootstrap al 95 %

| Modo | Métrica | Valor | IC 95 % |
|---|---|---:|---|
| `as_is` | Precision | 0.667 | [0.000, 1.000] |
| `as_is` | Recall | 0.250 | [0.000, 0.600] |
| `as_is` | F1 | 0.364 | [0.000, 0.667] |
| `as_is` | FPR | 0.083 | [0.000, 0.273] |
| `as_is` | PR-AUC | 0.875 | [0.614, 1.000] |
| `as_is` | ROC-AUC | 0.927 | [0.750, 1.000] |
| `stripped` | Precision | 0.500 | [0.000, 1.000] |
| `stripped` | Recall | 0.125 | [0.000, 0.400] |
| `stripped` | F1 | 0.200 | [0.000, 0.500] |
| `stripped` | FPR | 0.083 | [0.000, 0.273] |
| `stripped` | PR-AUC | 0.446 | [0.233, 0.733] |
| `stripped` | ROC-AUC | 0.578 | [0.417, 0.760] |

Con n=20 los intervalos son anchos y tratan las muestras como independientes. Se
reportan como referencia, no como precisión estimada.

### 2.3 Metadatos

| Modo | Auténticos con `METADATA_EDICION` | Manipulados con `METADATA_EDICION` |
|---|---:|---:|
| `as_is` | 0/12 | 8/8 |
| `stripped` | 0/12 | 0/8 |

Los metadatos separan perfectamente el lote **porque el generador usó Photoshop para
todos los manipulados**. Es un atajo del corpus, no una señal de contenido. Además,
`METADATA_EDICION` pesa 0.25 (severidad media): por sí solo no cruza el umbral 0.50.
Eso explica que los ocho manipulados tengan score 0.25 en `as_is` y solo dos crucen el
umbral al sumar otra señal.

### 2.4 Señal para revisión (no es el umbral operativo)

Para no confundir “no positivo” con “sin ninguna señal”, se cuenta también cuántas
muestras alcanzan al menos 0.25:

| Modo | Manipulados con score ≥ 0.25 | Auténticos con score ≥ 0.25 |
|---|---:|---:|
| `as_is` | 8/8 | 1/12 (M12) |
| `stripped` | 2/8 (M19, M20) | 1/12 (M12) |

Esta banda solo indica prioridad de revisión. El positivo de la tabla principal sigue
siendo `score ≥ 0.50`; no se cambia el umbral ni se presenta la banda media como una
probabilidad.

---

## 3. Antes y después de las correcciones

La corrida anterior a las correcciones de extracción e integración producía números
mejores, pero **no eran confiables**: varios “aciertos” dependían de errores de catálogo
y de cruce de actas, no de evidencia del ataque.

| Modo | Corrida | TP | TN | FP | FN | Precision | Recall | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `as_is` | previa | 7 | 6 | 6 | 1 | 0.538 | 0.875 | 0.667 | 0.500 |
| `as_is` | actual | 2 | 11 | 1 | 6 | 0.667 | 0.250 | 0.364 | 0.083 |
| `stripped` | previa | 6 | 6 | 6 | 2 | 0.500 | 0.750 | 0.600 | 0.500 |
| `stripped` | actual | 1 | 11 | 1 | 7 | 0.500 | 0.125 | 0.200 | 0.083 |

### 3.1 Qué se corrigió

1. **Catálogo tolerante a OCR.** El catálogo exigía coincidencia exacta. Variantes como
   `Computacion el Informâtica`, `Mecânica Automotriz` o `Ofimâtica yt herramientas
   digitales` generaban `PROGRAMA_DESCONOCIDO` / `MODULO_DESCONOCIDO`. Ahora se comparan
   sin tildes, con normalización y similitud ≥ 0.92. Los documentos auténticos M07–M11
   dejaron de ser falsos positivos.
2. **Resolución institucional.** La regla solo aceptaba `RD-...`; las plantillas usan
   `N.° 252-2026-DG-CETPRO-SA` y sus variantes OCR `N.O`, `N.`, `N.o`. M12 dejó de
   recibir `RESOLUCION_INVALIDA`.
3. **Actas con varios estudiantes.** El cruce usaba `estudiante_0` / `dni_0` como si
   fueran el titular. Eso producía `REGISTRO_NO_COINCIDENTE` en actas auténticas y en
   actas manipuladas por razones equivocadas. Ahora el cruce se omite (`score=None`)
   cuando no hay un titular único. Es correcto, pero también elimina la comparación de
   notas del acta: queda como limitación.
4. **Extracción de fechas y etiquetas.** Se aceptan fechas largas en español
   (`15 de febrero de 2026`, `29 DE SETIEMBRE DE 2026`) y los alias `Se expide` /
   `Expedido`; las correcciones previas de etiquetas compactas y geometría también
   entran en esta comparación. M03–M05 y M10–M11 dejaron de recibir `CAMPOS_FALTANTES`.
5. **Comparación difusa de programa/módulo en el registro.** El cruce ya no reporta
   diferencias por conectores o tildes del OCR; solo si el texto es realmente distinto.

### 3.2 Por qué caen los aciertos

Los verdaderos positivos previos M13–M16 y M18 estaban sostenidos por
`PROGRAMA_DESCONOCIDO` y/o `REGISTRO_NO_COINCIDENTE`, no por el ataque:

- M13 (T1) cambia una nota de 16 a 18; el registro no compara notas.
- M14 (T2) inserta un sello; copy-move no lo detecta.
- M15 (T3) duplica una firma; copy-move no la detecta.
- M16 (T4) cambia una nota de 09 a 14 en un acta multiestudiante; el cruce se omite.
- M18 (T6) es un documento fabricado con datos que coinciden con el registro; solo los
  metadatos lo delatan.

Al corregir los errores, esos documentos vuelven a su score real: 0.25 en `as_is`
(metadatos) y 0.00 en `stripped`. **La caída del recall es el resultado honesto**, no una
regresión del pipeline: antes se estaba midiendo el ruido de integración.

---

## 4. Detalle por muestra

| ID | Tipo | Etiqueta | `as_is` | pred | `stripped` | pred | Señales (`as_is` / `stripped`) |
|---|---|---|---:|---|---:|---|---|
| M01 | `dni` | auténtico | 0.00 | no | 0.00 | no | — / — |
| M02 | `dni` | auténtico | 0.00 | no | 0.00 | no | — / — |
| M03 | `certificado_estudios` | auténtico | 0.00 | no | 0.00 | no | — / — |
| M04 | `certificado_estudios` | auténtico | 0.00 | no | 0.00 | no | — / — |
| M05 | `partida_nacimiento` | auténtico | 0.00 | no | 0.00 | no | — / — |
| M06 | `ficha_matricula` | auténtico | 0.00 | no | 0.00 | no | — / — |
| M07 | `ficha_matricula` | auténtico | 0.00 | no | 0.00 | no | — / — |
| M08 | `acta_evaluacion` | auténtico | 0.00 | no | 0.00 | no | — / — |
| M09 | `acta_evaluacion` | auténtico | 0.00 | no | 0.00 | no | — / — |
| M10 | `certificado_modulo` | auténtico | 0.00 | no | 0.00 | no | — / — |
| M11 | `certificado_modulo` | auténtico | 0.00 | no | 0.00 | no | — / — |
| M12 | `titulo_auxiliar_tecnico` | auténtico | 0.60 | **sí** | 0.60 | **sí** | `CAMPOS_FALTANTES` / `CAMPOS_FALTANTES` |
| M13 | `certificado_modulo` | T1 | 0.25 | no | 0.00 | no | `METADATA_EDICION` / — |
| M14 | `certificado_estudios` | T2 | 0.25 | no | 0.00 | no | `METADATA_EDICION` / — |
| M15 | `acta_evaluacion` | T3 | 0.25 | no | 0.00 | no | `METADATA_EDICION` / — |
| M16 | `acta_evaluacion` | T4 | 0.25 | no | 0.00 | no | `METADATA_EDICION` / — |
| M17 | `dni` | T5 | 0.25 | no | 0.00 | no | `METADATA_EDICION` / — |
| M18 | `certificado_modulo` | T6 | 0.25 | no | 0.00 | no | `METADATA_EDICION` / — |
| M19 | `dni` | T7 | 0.85 | **sí** | 0.60 | **sí** | `REGISTRO_DIFERENCIA`, `METADATA_EDICION` / `REGISTRO_DIFERENCIA` |
| M20 | `ficha_matricula` | T8 | 0.50 | **sí** | 0.25 | no | `METADATA_EDICION`, `REGISTRO_DIFERENCIA` / `REGISTRO_DIFERENCIA` |

Lectura por ataque:

| Ataque | Muestra | Señal de contenido | Resultado |
|---|---|---|---|
| T1 — sustitución de texto | M13 | nota 16 → 18; el registro no guarda notas | no detectado en `stripped` |
| T2 — inserción de sello | M14 | copy-move = 0 | no detectado en `stripped` |
| T3 — firma duplicada | M15 | copy-move = 0 | no detectado en `stripped` |
| T4 — cambio de nota | M16 | acta multiestudiante; cruce omitido | no detectado en `stripped` |
| T5 — sustitución de fotografía | M17 | no hay forense de fotografía/rostro | no detectado en `stripped` |
| T6 — documento fabricado | M18 | campos coinciden con el registro | no detectado en `stripped` |
| T7 — sustitución de identidad | M19 | `REGISTRO_DIFERENCIA` de nombre | **detectado** |
| T8 — matrícula fuera de periodo | M20 | `REGISTRO_DIFERENCIA` de periodo (media) | señal media; cruza 0.50 solo con metadatos |

M20 merece una nota: en `stripped` **sí hay una señal** (periodo extraído `2026-I` frente
a `2025-I` del registro), pero pesa 0.25 y no cruza el umbral 0.50. La banda de la demo
lo muestra como “Medio”, no como “Bajo”. No es un falso negativo silencioso: es una
señal para revisión que el umbral estricto no cuenta como positivo.

---

## 5. Error residual conocido: M12

M12 es un título auténtico. La imagen dice `EXPEDIDO EL 29 DE SETIEMBRE DE 2026`, pero
docTR lee `291` (un `1` espurio) y la extracción no puede interpretar la fecha. El campo
`fecha_emision` queda ausente y `CAMPOS_FALTANTES` (severidad alta para títulos) produce
score 0.60.

No se corrige el `291` a `29` de forma silenciosa: sería inventar texto que el OCR no
leyó. El falso positivo se documenta y se deja para revisión humana. Una mejora futura
es distinguir “campo ausente” de “fecha ilegible” con una señal propia y un peso
justificado, no adivinar el dígito.

---

## 6. Límites

1. **Corpus sintético y pequeño.** 20 documentos, 12 personas y una muestra por ataque.
   No permite estimar generalización ni entrenar F4.
2. **Atajo de metadatos.** Los 8 manipulados llevan `Adobe Photoshop 25.0`; los 12
   auténticos, `CaptureOnTouch`. En `as_is` el detector “ve” el editor; en `stripped` no.
   La diferencia entre modos mide ese atajo, no una capacidad forense.
3. **Score no calibrado.** 0.50 es un umbral fijo de referencia. No es una probabilidad
   ni un punto operativo validado.
4. **Registro sintético.** `_personas.csv` es un espejo de 12 personas ficticias, no el
   registro institucional. No compara notas, firmas ni fotografías.
5. **Actas multiestudiante.** El cruce se omite correctamente para no comparar contra un
   titular falso, pero eso deja sin detector los cambios de nota dentro del acta.
6. **Forense local no operativo.** Copy-move no marca ninguna muestra; ELA, ruido,
   cromática y JPEG no entran al score. T2, T3 y T5 necesitan F4/F5.
7. **Sin evaluación humana.** No hay kappa ni revisión de expertos; no se puede afirmar
   que las bandas coincidan con el criterio institucional.
8. **IC bootstrap anchos.** Con n=20 y muestras no independientes, los intervalos son
   solo una referencia.

---

## 7. Reproducción

```bash
# Pipeline completo sobre las 20 muestras, ambos modos
make eval-samples

# Equivalente
uv run python -m fraude.eval.samples_eval data/muestras \
  --mode both \
  --out artifacts/eval/samples_pipeline.json \
  --today 2026-11-05
```

El comando requiere docTR y los modelos descargados (extra `ml`). El reporte incluye
métricas, IC bootstrap, evidencia por muestra, campos extraídos, señales forenses y
tiempos. `data/` y `artifacts/` están en `.gitignore`: no se commitean.

---

## 8. Qué sigue

El resultado correcto de F3 es este: **detecta inconsistencias de identidad y de
registro, y marca metadatos de edición, pero no detecta ataques locales de imagen ni
cambios académicos que el registro no modela**. Antes de F4:

- ampliar el corpus sintético con más personas y varias muestras por ataque;
- añadir un registro académico por estudiante/módulo que permita comparar notas y
  periodos sin inventar un titular único para actas;
- mejorar la forense local (copy-move, sellos, firmas, fotografía) y medirla contra
  auténticos;
- separar “campo ausente” de “fecha ilegible” con una señal explicable;
- mantener el umbral 0.50 y reportar también la banda de revisión (0.25–0.49) sin
  confundirla con una decisión;
- no entrenar F4 con una muestra por ataque.

Cualquier número de este documento debe citarse como **provisional, sobre un corpus
sintético de 20 documentos y con un score heurístico no calibrado**.
