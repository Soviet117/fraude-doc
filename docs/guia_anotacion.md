# Guía de anotación — Taxonomía T1–T8

**Versión:** 1.0
**Fase:** F1 (datos y terreno)
**Ámbito:** CETPRO San Agustín, Nueva Cajamarca
**Documento de referencia:** `plan-sistema-fraude-documental.md` §6.2, §6.3 y §6.4

Esta guía define **qué se etiqueta, cómo se etiqueta y cuándo una muestra es ambigua**.
Su objetivo es que dos anotadores produzcan las mismas etiquetas sobre el mismo documento
(acuerdo inter-anotador verificable) y que cada etiqueta sea explicable.

---

## 1. Principios

1. **Una etiqueta sin evidencia no existe.** Si no puedes señalar la región o describir la
   inconsistencia, no la etiquetes: marca `uncertain` y pide una segunda revisión.
2. **La calidad de imagen no es fraude.** Desenfoque, inclinación, recorte o brillo se
   registran como `quality_flags`, nunca como ataque.
3. **Se etiqueta lo que se ve, no lo que se sospecha.** El sistema prioriza revisión humana;
   la anotación debe poder defenderse ante un evaluador externo.
4. **Sin PII en las etiquetas.** Nombres, DNI, direcciones y fotografías se anonimizan antes
   de anotar o se mantienen en una copia restringida (ver §9).
5. **Todo es reproducible.** Cada etiqueta guarda anotador, fecha, versión de la guía y
   `document_id` (SHA-256 del archivo original).

---

## 2. Documentos del alcance

| `document_type` | Documento | Etapa |
|---|---|---|
| `dni` | Documento Nacional de Identidad | Admisión |
| `certificado_estudios` | Certificado de estudios | Admisión |
| `partida_nacimiento` | Partida de nacimiento | Admisión |
| `ficha_matricula` | Ficha de matrícula | Matrícula |
| `acta_evaluacion` | Acta de evaluación | Egreso |
| `certificado_modulo` | Certificado de módulo ocupacional | Egreso |
| `titulo_auxiliar_tecnico` | Título de Auxiliar Técnico | Titulación |

Cada documento puede tener **varias páginas**. La unidad de anotación es la página, pero el
`document_id` es el del archivo completo.

---

## 3. Esquema de etiqueta

Se usa un JSON por página. Los nombres de campos van en inglés (convención del proyecto);
los valores de texto libre, en español.

```json
{
  "schema_version": "1.0",
  "document_id": "sha256:...",
  "source_file": "raw/acta_evaluacion/2024/acta_001.pdf",
  "page": 1,
  "document_type": "acta_evaluacion",
  "is_authentic": false,
  "attack_types": ["T1"],
  "primary_attack": "T1",
  "quality_flags": {
    "capture_type": "scan",
    "blur": "low",
    "skew_deg": 0.8,
    "resolution_dpi": 300,
    "crop": false,
    "glare": false
  },
  "regions": [
    {
      "attack_type": "T1",
      "bbox": [120, 340, 210, 48],
      "mask_path": "annotations/masks/<document_id>/r001.png",
      "field": "nota_final",
      "original_value": "14",
      "altered_value": "18",
      "confidence": "alta",
      "notes": "Fuente distinta y trazo más grueso que el resto de la fila."
    }
  ],
  "semantic_flags": [],
  "uncertain": false,
  "annotator": "iniciales",
  "reviewed_by": null,
  "created_at": "2026-10-23"
}
```

Reglas del esquema:

- `bbox`: `[x, y, width, height]` en píxeles de la imagen original, origen arriba-izquierda.
- `mask_path`: PNG binario del mismo tamaño que la imagen; blanco (255) = región alterada.
  Obligatoria para T1–T5 cuando sea posible; para T6–T8 se permite solo etiqueta de documento.
- `field`: nombre del campo afectado (`nombre`, `dni`, `fecha_expedicion`, `nota_final`,
  `foto`, `sello`, `firma`, etc.).
- `original_value` / `altered_value`: cadenas de texto. Si el elemento no existía, usar
  `"ausente"`. Si no se puede recuperar el original, usar `"desconocido"`.
- `attack_types`: lista; un documento puede tener más de un ataque.
- `primary_attack`: el ataque más grave o el que mejor explica el caso (ver §6).
- `uncertain`: `true` cuando la evidencia no alcanza para decidir; exige segunda revisión.
- `semantic_flags`: lista de inconsistencias detectadas sin manipulación visual (fechas
  imposibles, DNI con dígito verificador inválido, curso inexistente, etc.).

---

## 4. Taxonomía T1–T8

### T1 — Sustitución de texto

**Definición:** se reemplaza el contenido de un campo (fecha, nombre, nota, código) por otro.

**Señales típicas:**
- Fuente, tamaño, grosor o espaciado distintos al resto de la línea.
- Línea base desalineada; caracteres que "flotan" o se superponen al fondo.
- Textura o ruido del fondo diferente alrededor del texto.
- Bordes de glifos con halos, restos de compresión o doble impresión.
- Tinta de distinta intensidad o color.

**Qué anotar:** `bbox` ajustado al texto alterado y `mask_path` de los glifos (dilatados
2–3 px). Registrar `field`, `original_value` y `altered_value`.

**Confusión frecuente:** tipeo distinto por una reimpresión legítima. Si no hay evidencia
visual de sustitución, no es T1; puede ser T8 si el dato es incoherente.

---

### T2 — Inserción de sello o firma

**Definición:** se agrega un sello o firma que no estaba en el documento original.

**Señales típicas:**
- Sello superpuesto al texto sin transparencia real o con transparencia uniforme.
- Bordes perfectamente nítidos, color plano, sin sangrado ni textura de tinta.
- Perspectiva o escala incompatible con el documento.
- Firma idéntica (copiada) en documentos distintos.

**Qué anotar:** `bbox` y máscara del elemento insertado. `original_value = "ausente"`.
Si el sello tapa texto, anotar también la región tapada en `notes`.

**Confusión frecuente:** sellos reales escaneados que se ven planos por la digitalización.
Si el sello pertenece al documento y solo está degradado por el escaneo, no es T2.

---

### T3 — Duplicación de sello/firma (copy-move)

**Definición:** un sello o firma ya presente se copia y se pega en otra parte del mismo
documento (o en otro documento de la misma persona).

**Señales típicas:**
- Dos elementos con forma, ruido y bordes idénticos.
- Copia exacta de píxeles, a veces con rotación o escala leve.
- Zona pegada con ruido distinto al fondo.

**Qué anotar:** máscara de la región **pegada** (la duplicada). Si se puede identificar el
origen, registrar su `bbox` en `notes` o en un campo `source_bbox` (opcional).

**Confusión frecuente:** sellos repetidos legítimamente (mismo sello estampado dos veces).
La repetición legítima tiene variación de tinta y posición; la copia digital es idéntica.

---

### T4 — Borrado/limpieza (inpainting)

**Definición:** se elimina contenido y se rellena la zona para ocultarlo.

**Señales típicas:**
- Zona anormalmente lisa o sin la textura del papel.
- Patrón repetido o manchas que no siguen el fondo.
- Bordes difusos, halos o desenfoque localizado.
- Ruido estadísticamente distinto al resto de la imagen.

**Qué anotar:** máscara de la región borrada/rellenada. Si se sabe qué se eliminó, registrar
`original_value`; si no, `"desconocido"`.

**Confusión frecuente:** manchas o correcciones reales del documento. Sin evidencia de
relleno digital, no es T4.

---

### T5 — Reemplazo de fotografía

**Definición:** se sustituye la fotografía del documento de identidad por otra.

**Señales típicas:**
- Borde de la foto con recorte, halo o pegamento digital.
- Iluminación, fondo o resolución incompatibles con el resto del documento.
- Perspectiva o escala de la cara inconsistente.
- Zona de la foto con ruido o compresión distinta.

**Qué anotar:** `bbox` y máscara del área de la fotografía reemplazada. `field = "foto"`.
No se anota el rostro como PII en la etiqueta exportada.

**Confusión frecuente:** fotos originales de baja calidad o reimpresas. La evidencia debe ser
el reemplazo, no la mala calidad.

---

### T6 — Documento completo fabricado

**Definición:** el documento entero es falso: plantilla, formato, sellos o datos no
corresponden a un documento emitido por el CETPRO.

**Señales típicas:**
- Plantilla, tipografía, logotipo o distribución distintos a los oficiales.
- Campos o secciones que no existen en el formato real.
- Ausencia de elementos de seguridad (firmas, sellos, códigos, numeración).
- Metadatos o fechas incompatibles con la emisión.

**Qué anotar:** etiqueta a nivel de documento (`attack_types = ["T6"]`), sin máscara
obligatoria. Si hay una región claramente falsa, se puede agregar `bbox` como apoyo.

**Confusión frecuente:** documento antiguo con formato distinto pero legítimo. Verificar
contra la galería de plantillas y el registro antes de etiquetar T6.

---

### T7 — Reutilización de documento de otra persona

**Definición:** se presenta un documento genuino que pertenece a otra persona (suplantación),
sin que el documento haya sido manipulado visualmente.

**Señales típicas:**
- El documento coincide con un registro existente de otra persona.
- Hash o embedding idéntico a un documento ya visto.
- Datos de identidad que no corresponden al solicitante.

**Qué anotar:** etiqueta a nivel de documento (`T7`), sin máscara. Registrar en `notes` la
evidencia de registro (ID de coincidencia, no PII). Requiere cruce con el registro
institucional; no se deduce solo de los píxeles.

**Confusión frecuente:** T5 (foto reemplazada) no es T7. Si además hay manipulación visual,
se etiqueta T5 y se registra T7 como `semantic_flag` si hay evidencia de registro.

---

### T8 — Inconsistencia semántica sin manipulación visual

**Definición:** el documento no muestra manipulación de píxeles, pero su contenido es
incoherente con el registro o consigo mismo.

**Señales típicas:**
- Fechas imposibles (egreso anterior al ingreso, edad incompatible con el grado).
- DNI con dígito verificador inválido o formato incorrecto.
- Curso o módulo que el CETPRO no ofrece.
- Notas fuera de escala o promedio incompatible con las notas.
- Nombre que no coincide entre campos del mismo documento.

**Qué anotar:** etiqueta a nivel de documento (`T8`) y, si aplica, `bbox` del campo
inconsistente. Registrar la regla violada en `semantic_flags`
(ej. `"fecha_egreso_anterior_ingreso"`).

**Confusión frecuente:** error administrativo real del CETPRO. Si el documento es auténtico
pero el dato es erróneo, sigue siendo T8 para el sistema (inconsistencia), aunque la causa no
sea fraude. Anotarlo en `notes`.

---

## 5. Resumen de anotación por tipo

| ID | Nivel | Máscara | Campo original/altered |
|---|---|---|---|
| T1 | Región | Obligatoria | Sí |
| T2 | Región | Obligatoria | Sí (`ausente`) |
| T3 | Región | Obligatoria | Parcial |
| T4 | Región | Obligatoria | Si se conoce |
| T5 | Región | Obligatoria | `foto` |
| T6 | Documento | Opcional | No |
| T7 | Documento | No | No |
| T8 | Documento/campo | Opcional | Sí, si aplica |

---

## 6. Reglas de decisión

1. **Multi-etiqueta:** si hay varios ataques, anotar todos en `attack_types` con regiones
   separadas. `primary_attack` es el más grave según este orden:
   `T6 > T7 > T5 > T3 > T2 > T1 > T4 > T8`.
2. **T6 manda a nivel de documento:** si el documento entero es fabricado, `primary_attack = T6`
   aunque también haya texto sustituido o sellos insertados. Las regiones visuales se anotan
   igual como evidencia secundaria.
3. **T7 no se deduce de píxeles:** solo se etiqueta con evidencia de registro o de identidad.
   Si no hay registro, no se etiqueta T7; se deja `uncertain`.
4. **T8 solo sin manipulación visual:** si hay evidencia visual, se etiqueta el tipo visual
   (T1–T5) y la inconsistencia semántica pasa a `semantic_flags`.
5. **Documento auténtico:** `is_authentic = true`, `attack_types = []`, `regions = []`.
   Un documento auténtico de mala calidad sigue siendo auténtico; se marcan `quality_flags`.
6. **Ambigüedad:** `uncertain = true` y revisión por un segundo anotador. No se fuerza una
   etiqueta para "completar" el dataset.

---

## 7. Procedimiento de anotación

1. **Preparar la copia anonimizada** del documento (ver §9) y calcular su `document_id`.
2. **Registrar metadatos** de captura: tipo (`scan`, `photo`, `pdf_render`), resolución,
   rotación aproximada.
3. **Revisar a 100 % y a 200–400 %** para buscar artefactos locales.
4. **Anotar la región** con bounding box; para T1–T5, generar la máscara binaria.
5. **Registrar el campo y los valores** original/altered cuando se conozcan.
6. **Marcar la etiqueta de documento** (`is_authentic`, `attack_types`, `primary_attack`).
7. **Guardar** el JSON y la máscara con la convención de rutas del §8.
8. **Segunda revisión** en los casos `uncertain` y en el 15 % de muestras aleatorias.

Herramienta prevista: **Label Studio** (E1-4 del backlog), con proyecto de detección de
objetos y campos personalizados. La guía no depende de la herramienta: el JSON es la
fuente de verdad.

---

## 8. Convención de rutas

```
data/
├── raw/                      # originales, nunca se modifican (fuera de git)
├── interim/                  # copias anonimizadas para anotar
└── annotations/
    ├── labels/<document_id>.json
    └── masks/<document_id>/r001.png
```

`data/` está en `.gitignore`. Las etiquetas y máscaras se sincronizan con `rsync` entre las
3 PCs, nunca por git.

---

## 9. Anonimización antes de anotar

- Sustituir nombres, DNI, direcciones y teléfonos por identificadores (`PERSONA_001`,
  `DNI_001`) en la copia de trabajo.
- La fotografía se conserva **solo** en la copia restringida necesaria para T5; en la copia
  compartida se reemplaza por un rectángulo neutro.
- La tabla de correspondencia (seudónimo → dato real) se guarda cifrada y separada de las
  imágenes; se destruye al cierre del proyecto.
- Las etiquetas exportadas no contienen PII: solo `document_id`, rutas de máscara y valores
  de campos cuando sean necesarios para el entrenamiento.
- Consentimiento institucional por escrito antes de recibir cualquier documento (§6.3).

---

## 10. Control de calidad

- **Doble anotación:** 15 % de las muestras, elegidas al azar por tipo documental y ataque.
- **Acuerdo:** kappa de Cohen ≥ 0,70 por tipo de ataque. Si no se alcanza, se revisa la guía
  y se re-anota el lote.
- **Adjudicación:** un tercer anotador resuelve discrepancias; la decisión se registra en
  `reviewed_by` y `notes`.
- **Auditoría:** cada lote debe poder reproducirse desde el manifiesto (F2) y las etiquetas
  versionadas.
- **Métricas de anotación:** se reportan cantidad por tipo documental, por ataque y tasa de
  `uncertain`; sirven de insumo para la data card v1.

---

## 11. Qué NO se anota como fraude

- Desenfoque, inclinación, recorte, brillo, sombras o dedos en la foto.
- Ruido de compresión JPEG, marcas de escáner o líneas de fax.
- Diferencias de formato por antigüedad del documento, si son legítimas.
- Errores de OCR del sistema (no son errores del documento).
- Sospechas sin evidencia visual ni de registro.

Todo lo anterior se registra en `quality_flags` o en `notes`, no en `attack_types`.

---

## 12. Versionado

| Versión | Cambio | Fecha |
|---|---|---|
| 1.0 | Versión inicial F1: taxonomía, esquema y reglas de decisión | 2026-10-23 |

Los cambios de esta guía se registran aquí y se reflejan en `schema_version`. Un cambio de
taxonomía requiere aprobación de ambos integrantes y actualización del plan (§6.2).

---

## 13. Pendientes de esta guía

- Añadir ejemplos visuales por ataque cuando existan las primeras muestras sintéticas (F2).
- Validar la taxonomía con el CETPRO y con los evaluadores expertos (F1/F8).
- Definir el formato exacto de importación a Label Studio cuando se instale la herramienta.
