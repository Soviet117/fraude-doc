# F6 — Rama semántica, duplicados y explicación

> **Estado:** cerrado experimental. Extracción real con docTR sobre el corpus
> v2, consistencia de entidades, detección de reutilización y explicación con
> plantilla (fallback) y LLM local opcional. La evaluación se hizo con el LLM
> apagado, como exige el plan.

## 1. Componentes

| Componente | Archivo | Qué hace |
|---|---|---|
| Normalización | `src/fraude/nlp/normalize.py` | Nombres canónicos, fechas y huella de contenido |
| Semántica | `src/fraude/nlp/semantic.py` | Consistencia de nombre/DNI, fecha de emisión plausible, reutilización |
| Duplicados | `src/fraude/nlp/duplicates.py` | SHA-256 de píxeles + dHash + similitud de campos |
| Explicación | `src/fraude/nlp/explain.py` | Plantilla determinista y Ollama local con fallback |
| Evaluación | `src/fraude/nlp/evaluate.py` | Exactitud por campo, duplicados, banderas y explicaciones |

Comandos:

```bash
make pipeline-v2   # ejecuta F3 sobre los 50 documentos base
make eval-f6       # extracción, duplicados, banderas y explicaciones
```

## 2. Resultados de extracción

- Documentos base: **50**
- Exactitud de tipo documental: **0.9600**
- La extracción usa OCR real (docTR 1.1.0), no los campos del manifiesto.

Campos con exactitud alta (muestra):

| Campo | n | Exactitud |
|---|---:|---:|
| fecha_emision | 28 | 1.000 |
| fecha_nacimiento | 26 | 1.000 |
| nombres / apellidos | 10 / 10 | 1.000 |
| numero (DNI) | 10 | 1.000 |
| periodo | 14 | 1.000 |
| promedio | 8 | 1.000 |
| dni (global) | 26 | 0.923 |
| estudiante | 24 | 0.917 |
| programa | 24 | 0.917 |
| modulo | 19 | 0.895 |
| nota / horas / resolucion | 5–6 | 0.800 |

Campos que fallan de forma sistemática:

| Campo | n | Exactitud | Lectura |
|---|---:|---:|---|
| director | 24 | 0.000 | La firma/etiqueta no se recupera con el esquema actual |
| turno | 8 | 0.000 | Se confunde con el periodo en la ficha |
| foto / sello_* | 3–4 | 0.000 | No son campos de texto; el OCR no los extrae |

Estos fallos **no se maquillan**: se documentan como límite de la extracción y
se excluyen de las reglas que dependen de ellos. La tabla completa está en
`artifacts/eval/f6_semantic.md`.

## 3. Duplicados (T7)

| Métrica | Valor |
|---|---:|
| Recall T7 | 1.000 |
| FPR auténticos | 0.000 |
| TP / FP / FN / TN | 3 / 0 / 0 / 25 |

**Advertencia:** en el corpus v2, T7 es una copia exacta de un documento
auténtico de otra persona. La detección por hash de píxeles es trivial y no
debe citarse como validación real de reutilización. Un caso real implicaría
reescaneo, recompresión o recorte; el índice ya soporta dHash + similitud de
texto, pero no se evaluó con esas transformaciones.

## 4. Banderas semánticas

| Ataque | Banderas observadas |
|---|---|
| T1 | document_reuse (1) |
| T2 | dni_mismatch (1), document_reuse (4) |
| T3 | dni_mismatch (1), document_reuse (3) |
| T4 | dni_mismatch (1), document_reuse (3) |
| T5 | document_reuse (3) |
| T7 | document_reuse (3) |
| T8 | document_reuse (2), implausible_emission_date (1) |
| Auténticos | dni_mismatch (3) |

Las banderas `document_reuse` en T1–T5 aparecen porque esas manipulaciones
derivan de un documento auténtico del índice; no significan que el ataque sea
T7. La fusión F7 aprende a combinarlas con las demás señales, pero la
explicación debe redactarse con cuidado (ver `docs/f7_fusion.md`).

## 5. Explicaciones

- No vacías: **50/50**
- Mencionan revisión humana: **50/50**
- Fuente: **plantilla** (el LLM está apagado por defecto)
- El LLM local se prueba con `--use-llm`; si Ollama no responde en 8 s, se usa
  la plantilla y se registra el motivo. El LLM nunca decide ni cambia el score.

## 6. Límites

1. Corpus sintético, 50 documentos y 10 personas.
2. El T7 evaluado es una copia exacta; no mide reutilización real.
3. `director` y `turno` fallan; las reglas que dependen de ellos son débiles.
4. La explicación por plantilla es correcta pero repetitiva; el LLM no se
   evaluó con jueces humanos.
5. No hay NER de spaCy en la ruta actual: la extracción es regex + etiquetas +
   priors. La evaluación por campo es la métrica que manda.
