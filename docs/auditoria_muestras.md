# Auditoría del corpus local de muestras

**Fase:** F3 (verificación de integración)  
**Estado:** corpus sintético auditado; **no** es validación institucional ni dataset de entrenamiento.  
**Fecha de corrida:** 2026-11-05  
**Artefacto:** `artifacts/eval/muestras_audit.json` (ignorado por git)  
**Comando:** `make audit-samples`

Este documento registra qué contiene `data/muestras`, qué verificó la auditoría y
por qué el lote **no alcanza para iniciar F4**. Los documentos son sintéticos:
`_generador.py` declara que no corresponden a personas reales ni validan en
RENIEC o MINEDU. Las identidades de `_personas.csv` son ficticias.

---

## 1. Composición

| Elemento | Valor |
|---|---|
| Documentos | 20 (M01–M20) |
| Auténticos | 12 |
| Manipulados | 8, uno por ataque T1–T8 |
| Personas | 12 |
| Tipos documentales | 7 |
| Resolución | 2481 × 3508 px (A4 a 300 dpi) |
| Formatos | PDF + PNG por muestra |
| Máscaras | 8 archivos; 6 con píxeles y 2 vacíos (T6 y T8) |

Distribución por tipo:

| Tipo | Muestras |
|---|---:|
| `acta_evaluacion` | 4 |
| `certificado_modulo` | 4 |
| `dni` | 4 |
| `certificado_estudios` | 3 |
| `ficha_matricula` | 3 |
| `partida_nacimiento` | 1 |
| `titulo_auxiliar_tecnico` | 1 |

La cobertura de ataques es completa, pero con **una sola muestra por ataque**.
T6 (documento fabricado) y T8 (inconsistencia semántica) no modifican píxeles:
su máscara existe pero está vacía y no tienen bbox. Eso es correcto para esos
ataques y no debe tratarse como error de etiquetado.

---

## 2. Integridad verificada

La auditoría revisa el índice `_indice.csv`, los archivos y las etiquetas. No
interpreta el contenido ni decide autenticidad.

| Verificación | Resultado |
|---|---|
| Columnas mínimas del índice | OK |
| IDs únicos | OK |
| Tipos documentales válidos | OK |
| Variantes y ataques válidos | OK |
| Personas presentes en `_personas.csv` | OK |
| Documentos padre existentes y auténticos | OK |
| PDF existentes y con cabecera `%PDF-` | OK |
| PNG legibles y en formato PNG | OK |
| SHA-256 de cada PNG contra el índice | 20/20 |
| Imágenes duplicadas por hash | 0 |
| Máscaras de ataques visuales con píxeles | 6/6 |
| Bbox dentro de la imagen | 6/6 |
| Bbox coincide con la máscara | 6/6 |
| Máscaras vacías en T6/T8 | aceptadas |
| Errores de integridad | 0 |
| Advertencias de integridad | 0 |

Conclusión: el corpus es internamente consistente y se puede usar para pruebas
de integración y revisión cualitativa.

---

## 3. Pipeline sin OCR

Se ejecutó `analyze_document(..., run_text=False)` sobre las 20 imágenes. Esto
activa calidad y forense clásico, pero **no OCR, extracción, reglas semánticas
ni cruce de registro**. Por eso todas las muestras incluyen el hallazgo
`TIPO_NO_IDENTIFICADO`: es esperado y no es una medición de desempeño.

| Señal | Auténticos (n=12) | Manipulados (n=8) |
|---|---:|---:|
| Calidad OK | 12 | 8 |
| `METADATA_EDICION` (score ≥ 0.5) | 0 | 8 |
| `COPY_MOVE` (score ≥ 0.5) | 0 | 0 |

### 3.1 Hallazgo principal: atajo de metadatos

Las 8 muestras manipuladas llevan metadatos de **Adobe Photoshop 25.0**; las 12
auténticas llevan metadatos de escaneo. Un clasificador trivial que solo mire
`Software` separaría perfectamente las clases de este lote.

**Eso no es detección de fraude.** Es un artefacto del generador. Si se entrenara
un modelo con estas imágenes sin controlar el metadata, el modelo podría aprender
el atajo y fallar en documentos reales o en manipulaciones sin metadatos. La
auditoría lo marca como advertencia `metadata_shortcut` y no calcula accuracy ni
F1 sobre esta separación.

### 3.2 Copy-move

El detector conservador no marcó ninguna muestra, incluida T3 (duplicación de
sello/firma). Es consistente con la limitación ya registrada en
[`f3_linea_base.md`](f3_linea_base.md): el detector evita falsos positivos de
encabezados y sacrifica sellos pequeños. No se debe concluir que T3 sea
indetectable; solo que esta versión no lo detecta en este lote.

### 3.3 Lo que no se midió

- OCR real ni extracción por campo.
- Reglas semánticas, cruce de registro o T7/T8 por contenido.
- Localización neuronal ni Pixel-F1/IoU.
- Calibración, Brier, ECE o umbrales operativos.
- Latencia representativa.

---

## 4. ¿Alcanza para F4?

**No.** El lote sirve para probar que el pipeline no se rompe y para revisar
señales, pero no para entrenar un clasificador visual ni para reportar métricas.

Razones:

1. **Volumen insuficiente:** 20 documentos y 12 personas. Una partición agrupada
   de train/val/test dejaría pocos grupos y un solo ejemplo por ataque.
2. **Desequilibrio extremo:** 12 auténticos frente a 8 manipulados, con un
   ejemplar por ataque. El recall por ataque sería anecdótico.
3. **Confusión de metadatos:** todas las manipulaciones comparten la misma firma
   de software. Sin un control explícito, el modelo aprendería el generador.
4. **Sin campos completos:** el índice y `_personas.csv` no bastan para evaluar
   reglas o extracción; se necesitaría OCR o un manifiesto de campos.
5. **Sin OCR:** docTR no está instalado en este host, así que no se puede medir
   la ruta de texto.
6. **Sin registro institucional:** el cruce de registro sigue siendo sintético.

Por tanto, **F4 no se inicia con este corpus**. Cualquier número de entrenamiento
obtenido aquí sería una demostración del generador, no evidencia del sistema.

---

## 5. Qué sí se puede hacer con estas muestras

- Smoke test del pipeline completo sobre 20 documentos y 7 tipos.
- Auditoría de integridad antes de usar cualquier lote nuevo.
- Revisión cualitativa de calidad y señales forenses.
- Prueba futura de localización usando las 6 máscaras visuales no vacías.
- Verificación de que T6/T8 se etiquetan sin máscara y no contaminan localización.

---

## 6. Qué falta para F4

- Más documentos sintéticos con **metadatos neutros o controlados** y variantes
  de captura; o documentos reales anonimizados del CETPRO.
- Particiones agrupadas por persona con cobertura suficiente por ataque.
- Un conjunto de calibración separado del de prueba.
- docTR instalado y OCR medido.
- Registro institucional autorizado.
- Un protocolo explícito para reportar el baseline de metadatos como control de
  fuga, no como resultado del modelo.

---

## 7. Reproducción

Desde la raíz del repo, con el entorno `uv` sincronizado:

```bash
# Auditoría de integridad + calidad/forense sin OCR
make audit-samples
# equivalente:
uv run python -m fraude.eval.corpus_audit data/muestras \
  --pipeline --today 2026-11-05 \
  --out artifacts/eval/muestras_audit.json

# Solo integridad (sin ejecutar el pipeline)
uv run python -m fraude.eval.corpus_audit data/muestras \
  --out artifacts/eval/muestras_audit.json

# Pruebas del auditor
uv run pytest -q tests/test_corpus_audit.py
```

`data/` y `artifacts/` están en `.gitignore`: ni las muestras ni el JSON se
commitean. El reporte se regenera con los comandos anteriores.
