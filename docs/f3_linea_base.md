# Línea base F3 — tabla provisional

**Fase:** F3 (línea base sin deep learning)  
**Estado:** cerrada de forma provisional; **no** es la tabla final del sistema ni una medición de OCR real.  
**Fecha de corrida:** 2026-10-29  
**Artefacto:** `artifacts/eval/f3_baseline_v1.json` (ignorado por git)  
**Semilla:** 42 · **Bootstrap:** 2000 remuestras · **Umbral fijo:** 0.50

Esta tabla es el piso que F4 debe superar. Se construyó con el dataset sintético v1
mientras el CETPRO no entrega documentos reales y docTR no está instalado en el host.
Los campos y el tipo documental se tomaron del manifiesto como **OCR oráculo**; por eso
la tabla mide reglas, registro, calidad y señales forenses, pero **no** mide OCR ni
extracción real.

---

## 1. Qué se midió

| Elemento | Valor |
|---|---|
| Dataset | Sintético v1 (`data/synthetic/v1`) |
| Validación | 42 muestras: 36 auténticas + 6 T4 |
| Test | 30 muestras: 18 auténticas + 6 T1 + 6 T2 |
| Campos | `fields_source = manifest_oracle` |
| Tipo documental | `doctype_source = manifest_oracle` |
| Registro | Espejo sintético de 14 personas (`data/registry/synthetic.jsonl`) |
| Forense | Calculado sobre la imagen (ELA, ruido, cromática, JPEG, metadatos, copy-move) |
| Calidad | Calculada sobre la imagen |
| OCR | No ejecutado; docTR no está instalado en este host |

**No se midió:** OCR real, extracción real, T3–T8 (no están en test), documentos reales,
calibración, latencia ni comportamiento en producción.

---

## 2. Cómo se calcula el score

El score es una suma explicable de pesos por severidad, truncada a 1.0:

| Severidad | Peso |
|---|---:|
| Alta (`REGISTRO_DIFERENCIA`, `REGISTRO_NO_COINCIDENTE`, `COPY_MOVE`, fechas inválidas/futuras, DNI inválido, etc.) | 0.60 |
| Media (`CAMPOS_FALTANTES` en ficha/acta, `OCR_BAJO`, `METADATA_EDICION`, diferencias académicas de registro) | 0.25 |
| Baja (`CALIDAD_BAJA`) | 0.05 |

Ejemplos observados:

- T1: `REGISTRO_DIFERENCIA` alta → **0.60**.
- T4: `CAMPOS_FALTANTES` media → **0.25**.
- Auténtico con rotación: `CALIDAD_BAJA` baja → **0.05**.
- T1 con rotación: alta + baja → **0.65**.

El score **no es una probabilidad**: no está calibrado. La calibración corresponde a F6.
Las señales `noise` y `chromatic` saturan en auténticos sintéticos y no se usan en el score;
`ELA` y `JPEG` quedan como exploratorias. `copy-move` y `metadata` entran solo cuando el
motor de reglas las convierte en hallazgo.

---

## 3. Protocolo de umbrales

| Umbral | Valor | Uso |
|---|---:|---|
| Fijo | 0.50 | Referencia obligatoria del plan §7.4 |
| Elegido en validación | 0.25 | Mejor F1 en `val`; desempate cerca de 0.5 |
| Oráculo en test | 0.50 | Cota superior; **no** es un punto operativo válido |

Test se evaluó una sola vez. El umbral 0.25 se eligió únicamente con validación; el oráculo
se reporta para mostrar cuánto se podría ganar si se ajustara con test, pero no se usa como
resultado del sistema.

---

## 4. Resultados

### 4.1 Tabla principal

| Split / umbral | Umbral | TP | TN | FP | FN | Precision | Recall | F1 | FPR | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val / fijo | 0.50 | 0 | 36 | 0 | 6 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| val / elegido | 0.25 | 6 | 36 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| **test / fijo** | **0.50** | **6** | **18** | **0** | **6** | **1.000** | **0.500** | **0.667** | **0.000** | **0.7315** | **0.7639** |
| test / elegido | 0.25 | 6 | 18 | 0 | 6 | 1.000 | 0.500 | 0.667 | 0.000 | 0.7315 | 0.7639 |
| test / oráculo | 0.50 | 6 | 18 | 0 | 6 | 1.000 | 0.500 | 0.667 | 0.000 | 0.7315 | 0.7639 |

Lectura de la tabla:

- En **validación**, el umbral 0.50 no detecta T4 porque `CAMPOS_FALTANTES` pesa 0.25.
  El umbral elegido (0.25) sí lo detecta, sin falsos positivos en los 36 auténticos.
- En **test**, T1 se detecta por diferencia de registro; T2 no se detecta. El cambio de
  umbral no altera test porque T1 puntúa 0.60 y T2 puntúa 0.
- PR-AUC y ROC-AUC de validación son 1.0 porque el ranking separa perfectamente T4 de los
  auténticos; no implican que el umbral 0.50 sea bueno.

### 4.2 Intervalos bootstrap al 95 % (test, umbral 0.50)

| Métrica | Valor | IC 95 % |
|---|---:|---|
| Precision | 1.000 | [1.000, 1.000] |
| Recall | 0.500 | [0.222, 0.800] |
| F1 | 0.667 | [0.364, 0.889] |
| FPR | 0.000 | [0.000, 0.000] |
| PR-AUC | 0.7315 | [0.5055, 0.9000] |
| ROC-AUC | 0.7639 | [0.6040, 0.9201] |

Los intervalos son **optimistas**: el test contiene un solo grupo (`persona_0001`) y sus
30 muestras son variantes del mismo documento base y de sus ataques. El bootstrap
remuestrea muestras, no personas; con un solo grupo no puede estimar variabilidad entre
personas. No se debe leer “precision = 1.000” como evidencia de precisión perfecta.

### 4.3 Desglose por ataque (test, umbral 0.50)

| Ataque | n | TP | FN | Recall | Score medio |
|---|---:|---:|---:|---:|---:|
| T1 — sustitución de texto | 6 | 6 | 0 | 1.000 | 0.6083 |
| T2 — inserción de sello | 6 | 0 | 6 | 0.000 | 0.0083 |
| Auténticos | 18 | 0 FP | — | FPR 0.000 | 0.0056 |

T2 no modifica los campos ni el registro: inserta un sello en la imagen. Las reglas actuales
no lo ven y el detector de copy-move conservador tampoco lo marca. Es la limitación
principal de esta línea base.

En validación, con umbral 0.25, T4 alcanza recall 1.000 (6/6) y los 36 auténticos quedan
en FPR 0.000.

### 4.4 Desglose por tipo documental (test, umbral 0.50)

| Tipo | n | Positivos | TP | FN | FP | TN | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `certificado_estudios` | 18 | 12 | 6 | 6 | 0 | 6 | 0.500 | 0.000 |
| `ficha_matricula` | 6 | 0 | 0 | 0 | 0 | 6 | — | 0.000 |
| `partida_nacimiento` | 6 | 0 | 0 | 0 | 0 | 6 | — | 0.000 |

`ficha_matricula` y `partida_nacimiento` solo tienen muestras auténticas en test; su recall
no es interpretable. El único tipo con positivos es `certificado_estudios`, y su recall es
0.500 porque incluye T1 (detectado) y T2 (no detectado).

---

## 5. Limitaciones

1. **OCR oráculo.** Los campos y el tipo documental vienen del manifiesto, no de docTR.
   La tabla no mide OCR ni extracción; un OCR real puede fallar de otra forma.
2. **Dataset sintético.** No sustituye documentos reales del CETPRO. Sirve como piso
   reproducible, no como estimación de desempeño institucional.
3. **Un solo grupo en test.** Las 30 muestras de test son de `persona_0001`; los IC
   bootstrap son optimistas y no capturan variabilidad entre personas.
4. **Cobertura de ataques.** Test v1 solo tiene T1 y T2. Validación solo tiene T4. No hay
   recall para T3, T5, T6, T7 ni T8 en esta tabla.
5. **T2 no detectado.** La inserción de sello no cambia campos ni registro; requiere
   forense local, comparación de plantilla o un modelo entrenado (F4).
6. **T3 subdetectado.** El detector de copy-move se hizo conservador para evitar falsos
   positivos de encabezados; un sello pequeño puede no reportarse.
7. **Señales descartadas.** `noise` y `chromatic` saturan en auténticos sintéticos; `ELA`
   y `JPEG` no se usan en el score. No son evidencia suficiente en esta versión.
8. **Sin calibración.** El score es heurístico; no es una probabilidad. Brier, ECE y
   curva de fiabilidad quedan para F6.
9. **Registro sintético.** El espejo tiene 14 personas y se construyó desde el manifiesto;
   no es el registro institucional ni valida identidad contra RENIEC.
10. **Umbral oráculo.** El umbral 0.50 coincide con el fijo en test, pero el oráculo no es
    un punto operativo: se calculó sobre test y no se debe usar para desplegar.

---

## 6. Reproducción

Desde la raíz del repo, con el entorno `uv` sincronizado:

```bash
# 1. Dataset sintético v1 (si no existe)
make synth
# equivalente:
# uv run python -m fraude.synth.build --out data/synthetic/v1 --count 42 --identities 14

# 2. Registro espejo sintético
uv run python -m fraude.registry.build \
  --manifest data/synthetic/v1/manifest.jsonl \
  --out data/registry/synthetic.jsonl

# 3. Línea base F3
uv run python -m fraude.eval.baseline \
  --manifest data/synthetic/v1/split_manifest.jsonl \
  --registry data/registry/synthetic.jsonl \
  --out artifacts/eval/f3_baseline_v1.json \
  --n-boot 2000 --seed 42 --today 2026-10-29

# 4. Pruebas del arnés
uv run pytest -q tests/test_eval.py
```

`data/` y `artifacts/` están en `.gitignore`: los datos y el JSON no se commitean. El
manifiesto y la semilla reconstruyen el dataset; el JSON de evaluación se regenera con el
comando anterior.

---

## 7. Qué sigue

F3 queda cerrada como línea base provisional. **No se inicia F4** hasta contar con:

- documentos reales del CETPRO, anonimizados y con acuerdo de uso;
- docTR instalado y OCR medido sobre esos documentos;
- más personas/grupos en test para que los intervalos bootstrap sean informativos;
- muestras de T3–T8 en test para medir lo que esta tabla no mide;
- registro institucional (o una versión autorizada) para el cruce;
- protocolo de calibración de §7.4 antes de reportar probabilidades.

Mientras eso no exista, cualquier número de esta tabla debe citarse como **provisional,
sobre datos sintéticos y con campos oráculo**.
