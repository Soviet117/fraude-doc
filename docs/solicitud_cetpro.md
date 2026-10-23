# Solicitud de acceso a documentos — CETPRO San Agustín

**Versión:** 1.0
**Fase:** F1 (datos y terreno)
**Referencia:** `plan-sistema-fraude-documental.md` §6.1 y §6.3

> Plantilla institucional. Completar los campos entre corchetes antes de imprimir o enviar.
> No incluye datos personales reales.

---

## 1. Datos del destinatario

- **Institución:** CETPRO San Agustín — Nueva Cajamarca
- **Destinatario:** [Nombre del director / responsable]
- **Cargo:** [Director(a) / Secretario(a) académico(a)]
- **Dirección:** [Dirección institucional]

## 2. Datos de los solicitantes

- **Proyecto:** Sistema de detección de fraude documental en documentos académicos del CETPRO San Agustín
- **Integrantes:** [Nombre 1], [Nombre 2]
- **Programa:** Ingeniería de Sistemas — [Universidad]
- **Ciclo:** 8.º ciclo
- **Correo de contacto:** [correo institucional]
- **Teléfono:** [teléfono]

## 3. Objeto de la solicitud

Solicitamos autorización para acceder a una muestra de documentos académicos emitidos o
recibidos por el CETPRO San Agustín, con el fin exclusivo de desarrollar y evaluar un
prototipo de investigación que detecte posibles manipulaciones documentales y priorice la
revisión humana.

El prototipo **no emite decisiones automáticas**, no valida oficialmente documentos ante
RENIEC ni MINEDU y no reemplaza los procedimientos institucionales. Su alcance es académico
y experimental.

## 4. Documentos solicitados

| Tipo de documento | Uso en el proyecto | Cantidad deseada |
|---|---|---|
| DNI (copia) | Documento de identidad | 30 |
| Certificado de estudios | Admisión | 30 |
| Partida de nacimiento | Admisión | 30 |
| Ficha de matrícula | Matrícula | 50 |
| Acta de evaluación | Egreso | 50 |
| Certificado de módulo ocupacional | Egreso | 50 |
| Título de Auxiliar Técnico | Titulación | 30 |
| Formatos en blanco (plantillas) | Referencia de autenticidad | 5–7 tipos |

**Muestra inicial:** 50 documentos variados para arrancar la anotación (F1).
**Objetivo:** hasta 300 documentos si la institución puede facilitarlos.
**Formatos preferidos:** PDF escaneado, PNG o TIFF sin pérdida; se aceptan JPEG si es el
único formato disponible.
**Periodo:** [años o rango de años] — completar según disponibilidad institucional.

No se solicitan documentos originales físicos: basta una copia digital.

## 5. Compromisos de tratamiento de datos

1. **Consentimiento y finalidad:** los documentos se usarán únicamente para este proyecto de
   investigación, sin fines comerciales.
2. **Anonimización:** antes de cualquier procesamiento se reemplazan nombres, DNI,
   direcciones, teléfonos y fotografías por identificadores. La tabla de correspondencia se
   guarda cifrada y separada de las imágenes.
3. **Confidencialidad:** los datos crudos permanecen en los equipos locales del equipo de
   investigación. **No se envían a APIs externas ni a servicios en la nube** (OpenAI, Google,
   Anthropic u otros).
4. **Acceso restringido:** solo los dos integrantes del proyecto acceden a los documentos.
5. **Ley N.° 29733:** se cumple la normativa peruana de protección de datos personales y su
   reglamento.
6. **Borrado:** al cierre del proyecto se eliminan los datos crudos y la tabla de
   correspondencia, salvo que la institución autorice su conservación.
7. **Sin publicación de PII:** los resultados, figuras y anexos del informe no incluyen
   documentos legibles ni datos personales.
8. **Trazabilidad:** cada documento recibe un identificador SHA-256 y un registro de acceso.

## 6. Procedimiento propuesto

1. La institución designa un responsable y entrega una copia digital en un medio local
   (USB o carpeta institucional).
2. El equipo calcula el hash de cada archivo y registra el inventario.
3. Se genera una copia anonimizada para anotación; el original queda en cuarentena.
4. Se anotan las muestras según la guía T1–T8 y se reportan resultados agregados.
5. Al finalizar, se presenta un informe breve a la institución y se eliminan los datos según
   el compromiso §5.6.

## 7. Consentimiento informado

Se solicita a la institución firmar el siguiente consentimiento:

> El CETPRO San Agustín autoriza a [Nombre 1] y [Nombre 2], integrantes del proyecto
> [nombre del proyecto], a utilizar una copia digital de los documentos detallados en el §4
> con fines exclusivamente académicos y de investigación. La institución conoce y acepta los
> compromisos de anonimización, confidencialidad y borrado descritos en el §5.

- **Nombre y cargo:** ______________________________
- **Firma:** ______________________________
- **Fecha:** ______________________________

## 8. Firmas de los solicitantes

- [Nombre 1] — ______________________________
- [Nombre 2] — ______________________________
- **Fecha:** ______________________________

## 9. Anexos

- Guía de anotación T1–T8 (`docs/guia_anotacion.md`).
- Resumen del plan del proyecto (`plan-sistema-fraude-documental.md` §0–§1).
- Inventario documental (se completa al recibir la muestra).
