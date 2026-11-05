# OCR real sobre el corpus local de muestras

**Fase:** F3 (medición de OCR y extracción)  
**Estado:** medición sobre corpus sintético; **no** es validación institucional.  
**Fecha de corrida:** 2026-11-05  
**Artefacto:** `artifacts/eval/ocr_samples.json` (ignorado por git)  
**Textos OCR:** `artifacts/ocr/samples/` (ignorados por git)  
**Comando:** `make ocr-samples`

Este documento registra la primera medición de OCR real con docTR sobre las 20
muestras de `data/muestras`. A diferencia de la tabla F3 principal, aquí **no se
usan campos oráculo**: el tipo documental y los campos salen del texto que docTR
reconoce en el PNG. El corpus sigue siendo sintético y pequeño; los números no
estiman desempeño institucional.

---

## 1. Qué se midió

| Elemento | Valor |
|---|---|
| Motor OCR | docTR 1.1.0 (`fast_base` + `crnn_vgg16_bn`) |
| Dispositivo | CPU (este host no tiene CUDA) |
| Muestras | 20 (12 auténticas + 8 manipuladas T1–T8) |
| Personas | 12 |
| Tipos documentales | 7 |
| Referencia de valores | `_personas.csv` e `_indice.csv` |

Métricas:

- **Recall de campos OCR:** proporción de valores esperados (DNI, nombres,
  fechas, programa, módulo, periodo) que aparecen en el texto reconocido.
- **Accuracy de tipo documental:** comparación entre el tipo del índice y el que
  detecta `extract_page` a partir del texto OCR.
- **Accuracy de extracción:** proporción de campos esperados que el extractor
  devuelve con el valor correcto.
- **Tokens alterados:** en T1 y T4 se comprueba si el valor manipulado sigue
  presente en el texto; no es una métrica de detección de fraude.

No se calcula CER/WER porque no existe una transcripción completa de referencia.
La partida de nacimiento usa fecha larga y el extractor solo reconoce
`dd/mm/aaaa`; su accuracy de extracción se reporta como `null`.

---

## 2. Resultado global

| Métrica | Antes de las correcciones | Ahora |
|---|---:|---:|
| Recall de campos OCR | 0.7576 | **0.9242** |
| Accuracy de tipo documental | 0.5000 | **1.0000** |
| Accuracy de extracción de campos | 0.0769 | **0.9231** |
| Tokens alterados presentes (T1/T4) | 2/2 | **2/2** |
| Confianza media OCR | 0.8600 | 0.8600 |
| Palabras medias | 203.95 | 203.95 |
| Latencia p50 | 5285.56 ms | 5422.21 ms |
| Latencia p95 | 7005.81 ms | 7295.90 ms |
| Warmup | 13407.76 ms | 14003.70 ms |

La confianza y el número de palabras no cambiaron porque el motor OCR es el
mismo. Lo que cambió fue la interpretación del texto: detección del tipo por el
título más temprano, etiquetas tolerantes a palabras pegadas por OCR, dos pasadas
de búsqueda y normalización de periodos. La latencia en CPU no representa al
laboratorio con GPU.

---

## 3. Desglose por tipo documental

| Tipo | n | Recall OCR | Tipo | Extracción | Confianza |
|---|---:|---:|---:|---:|---:|
| `acta_evaluacion` | 4 | 0.8333 | 1.0000 | 0.8333 | 0.8769 |
| `certificado_estudios` | 3 | 1.0000 | 1.0000 | 1.0000 | 0.8900 |
| `certificado_modulo` | 4 | 1.0000 | 1.0000 | 1.0000 | 0.8673 |
| `dni` | 4 | 0.9375 | 1.0000 | 0.9375 | 0.8204 |
| `ficha_matricula` | 3 | 0.8571 | 1.0000 | 0.8571 | 0.8595 |
| `partida_nacimiento` | 1 | 1.0000 | 1.0000 | — | 0.8645 |
| `titulo_auxiliar_tecnico` | 1 | 1.0000 | 1.0000 | 1.0000 | 0.8824 |

---

## 4. Desglose por ataque

| Grupo | n | Recall OCR | Tipo | Extracción |
|---|---:|---:|---:|---:|
| Auténticos | 12 | 0.9211 | 1.0000 | 0.9189 |
| T1 | 1 | 1.0000 | 1.0000 | 1.0000 |
| T2 | 1 | 1.0000 | 1.0000 | 1.0000 |
| T3 | 1 | 1.0000 | 1.0000 | 1.0000 |
| T4 | 1 | 0.6667 | 1.0000 | 0.6667 |
| T5 | 1 | 1.0000 | 1.0000 | 1.0000 |
| T6 | 1 | 1.0000 | 1.0000 | 1.0000 |
| T7 | 1 | 0.7500 | 1.0000 | 0.7500 |
| T8 | 1 | 1.0000 | 1.0000 | 1.0000 |

Cada ataque tiene una sola muestra: estas cifras describen el lote, no un recall
por ataque.

---

## 5. Errores que quedan y por qué no se maquillan

Los cuatro fallos restantes son errores reales de OCR en palabras concretas. El
arnés los conserva para no inflar la métrica con coincidencias difusas:

| Muestra | Campo | Esperado | OCR / extraído |
|---|---|---|---|
| M07 | `programa` | Computación e Informática | `Computacion el Informâtica` |
| M07 | `modulo` | Ofimática y herramientas digitales | `Ofimâtica yt herramientas digitales` |
| M09 | `modulo` | Automatización industrial | `Automatizacioni industrial` |
| M16 | `programa` | Electrónica Industrial | `Electrônical Industrial` |
| M19 | `nombres` | PATRICIA SOFÍA | `PATRICIAS SOFIA` |

La comparación ignora espacios perdidos (`BAUTISTAHUAMAN` se acepta frente a
`BAUTISTA HUAMÁN`), pero no letras extra: `PATRICIAS` sigue siendo un error
frente a `PATRICIA`. Los errores de M07, M09 y M16 son confusiones del motor
(`e`→`el`, `y`→`yt`, `i`/`l` extra) y no se corrigen con reglas del extractor.

M19 conserva además el apellido como `BAUTISTAHUAMAN`; la comparación compacta
lo acepta porque el OCR perdió el espacio, no una letra.

---

## 6. Cambios que produjeron la mejora

1. **Tipo documental por título más temprano.** Antes ganaba la primera palabra
   clave de una lista, aunque estuviera en el cuerpo. Ahora se busca la primera
   aparición de cada patrón en el texto; el título del encabezado gana a
   etiquetas como `DOCUMENTO NACIONAL DE IDENTIDAD` o `acta de evaluacion`.
2. **Etiquetas compactas.** docTR pega palabras (`FECHADEI NACIMIENTO`,
   `MODULOMATRICULADO`). Las etiquetas se comparan sin espacios, tildes ni
   puntuación, y el valor se recorta del texto original.
3. **Dos pasadas.** Primero se busca el valor en la misma línea; solo después en
   las líneas siguientes. Así una etiqueta específica no pierde frente a una
   genérica que apunta a una tabla.
4. **Periodos normalizados.** `2024-1`, `2024-I` y `2024-11` se unifican a
   `2024-I`/`2024-II` antes de comparar.
5. **Geometría de la ficha.** Programa y módulo comparten fila de dos columnas.
   Si el OCR devuelve una sola línea, la posición x del encabezado derecho corta
   la fila. Si la geometría falla y ambos campos contienen la misma fila, se
   omiten antes que duplicarlos.

---

## 7. Límites

1. **Corpus sintético.** 20 documentos y 12 personas no sustituyen documentos
   reales del CETPRO ni estiman desempeño institucional.
2. **Una muestra por ataque.** No hay recall por ataque defendible.
3. **Sin transcripción completa.** No se reporta CER/WER; el recall mide
   presencia de valores conocidos, no calidad global del texto.
4. **Partida de nacimiento.** Su fecha larga no se evalúa con el extractor
   actual; el `null` es deliberado.
5. **Latencia de CPU.** El host no tiene CUDA; los tiempos no representan al
   laboratorio con GPU.
6. **Metadatos.** El atajo de Photoshop del generador sigue vigente (8/8
   manipulados vs 0/12 auténticos). No se usa como métrica de detección.
7. **Sin validación humana.** No hay evaluadores expertos ni kappa en este lote.

---

## 8. Reproducción

```bash
# Requiere la extra ml instalada (docTR + torch)
uv sync --extra ml

make ocr-samples
# equivalente:
# uv run python -m fraude.eval.ocr_samples data/muestras \
#   --out artifacts/eval/ocr_samples.json \
#   --text-dir artifacts/ocr/samples
```

`data/` y `artifacts/` están en `.gitignore`: los PNG, los textos OCR y el JSON
no se commitean. El reporte se regenera con el comando anterior.
