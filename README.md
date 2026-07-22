# ConsumerComplaintCFPB
Financial complaint classification using SEC-BERT with confidence-based abstention


# Financial Complaint Classification with SEC-BERT

## Descripción

Este repositorio contiene un prototipo de clasificación automática de reclamaciones financieras a partir del texto de la queja.

El objetivo es predecir la categoría `Product` utilizando la columna `Consumer complaint narrative` del Consumer Complaint Database del Consumer Financial Protection Bureau (CFPB).

La solución propuesta consiste en realizar fine-tuning de [`nlpaueb/sec-bert-base`](https://huggingface.co/nlpaueb/sec-bert-base), un modelo BERT previamente entrenado sobre textos financieros.

El notebook incluye el flujo completo:

- Preparación y limpieza de los datos.
- Agrupación de las categorías originales.
- División estratificada en train, validation y test.
- Balanceo moderado del conjunto de entrenamiento.
- Fine-tuning de SEC-BERT.
- Evaluación global y por producto.
- Selección de un threshold de confianza.
- Matriz de confusión y análisis de errores.
- Ejemplo de inferencia sobre una nueva reclamación.

## Estructura del repositorio

```text
.
├── SEC_BERT.ipynb
├── README.md
└── .gitignore
```

El archivo `SEC_BERT.ipynb` contiene la solución completa, ejecutada y comentada.

El dataset y los checkpoints del modelo no se incluyen en el repositorio debido a su tamaño.

## Dataset

Se utiliza un corte fijo del Consumer Complaint Database del CFPB con reclamaciones:

- Recibidas entre el 1 de junio de 2023 y el 31 de mayo de 2024.
- Con narrativa disponible.
- Aproximadamente 582.000 observaciones.

El dataset puede descargarse directamente desde:

```text
https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?date_received_min=2023-06-01&date_received_max=2024-05-31&has_narrative=true&format=csv&no_aggs=true
```

Las variables utilizadas son:

- `Consumer complaint narrative`: texto de entrada.
- `Product`: categoría que se desea predecir.

### Preparación del archivo

Después de descargar el CSV:

1. Crear una carpeta llamada `data`.
2. Guardar el archivo como:

```text
data/complaints.csv
```

El notebook utiliza la siguiente ruta:

```python
df = pd.read_csv("data/complaints.csv")
```

El CSV no debe subirse al repositorio.


## Categorías utilizadas

Las categorías originales se agrupan en nueve familias de producto para unificar etiquetas históricas o equivalentes:

1. `Checking or savings account`
2. `Credit or prepaid card`
3. `Credit reporting`
4. `Debt collection or management`
5. `Money transfer, virtual currency, or money service`
6. `Mortgage`
7. `Payday, title, personal or advance loan`
8. `Student loan`
9. `Vehicle loan or lease`

Después de eliminar registros sin texto o sin producto, el dataset final contiene 582.490 reclamaciones.

La distribución presenta un fuerte desbalanceo. `Credit reporting` concentra aproximadamente el 72,4 % de las observaciones, mientras que varias categorías minoritarias representan menos del 1,5 %.

Por este motivo, el rendimiento no se evalúa únicamente mediante accuracy. También se utilizan macro F1, weighted F1 y métricas por producto.

## División de los datos

La división se realiza de forma estratificada para conservar la distribución de categorías:

| Conjunto | Porcentaje | Observaciones |
|---|---:|---:|
| Train | 80 % | 465.992 |
| Validation | 10 % | 58.249 |
| Test | 10 % | 58.249 |

El conjunto de validation se utiliza para evaluar el modelo y seleccionar el threshold de confianza.

El conjunto de test se mantiene separado hasta la evaluación final.


## Balanceo del conjunto de entrenamiento

El dataset original presenta un desbalanceo elevado. Para reducir el dominio de las categorías mayoritarias, se limita a 20.000 el número máximo de ejemplos por categoría únicamente en train.

Después del balanceo, la muestra efectiva de entrenamiento contiene 113.192 observaciones.

La estrategia utilizada es:

- Las categorías con más de 20.000 observaciones se submuestrean.
- Las categorías minoritarias conservan todos sus ejemplos.
- Validation y test mantienen la distribución original.

Esto permite mejorar la representación de las clases minoritarias durante el entrenamiento sin alterar el escenario real de evaluación.


## Modelo

El modelo base utilizado es:

```text
nlpaueb/sec-bert-base
```

SEC-BERT aporta representaciones lingüísticas previamente aprendidas sobre documentos financieros.

La capa final de clasificación se inicializa específicamente para las nueve familias de producto y se entrena junto con el resto del modelo.


## Tokenización

Las reclamaciones se tokenizan con una longitud máxima de 128 tokens.

- Las reclamaciones más largas se truncan.
- El padding se aplica dinámicamente en cada batch.
- Esta configuración reduce el consumo de memoria y el tiempo de entrenamiento.

Una longitud mayor podría conservar más información de las reclamaciones extensas, pero también aumentaría el coste computacional.


## Configuración del entrenamiento

| Parámetro | Valor |
|---|---:|
| Épocas | 1 |
| Learning rate | `2e-5` |
| Batch size de entrenamiento | 16 |
| Batch size de evaluación | 32 |
| Weight decay | `0.01` |
| Longitud máxima | 128 tokens |
| Métrica principal | Macro F1 |
| Precisión mixta | FP16 con GPU |

El fine-tuning se realizó durante una única época debido a las limitaciones de GPU disponibles durante el desarrollo del prototipo.

Con mayor capacidad de cómputo sería recomendable:

- Evaluar un número superior de épocas.
- Probar longitudes máximas de 256 y 512 tokens.
- Monitorizar las curvas de training loss y validation loss.
- Incorporar early stopping.
- Detener el entrenamiento cuando la métrica de validación deje de mejorar.

Los resultados deben interpretarse como los de un prototipo funcional y no como el máximo rendimiento alcanzable por el modelo.


## Resultados globales

El rendimiento final sobre el conjunto de test es:

| Métrica | Resultado |
|---|---:|
| Accuracy | 86,33 % |
| Weighted F1 | 87,20 % |
| Macro F1 | 73,42 % |
| Macro recall | 81,24 % |

La diferencia entre weighted F1 y macro F1 refleja que el rendimiento continúa siendo superior en las categorías más frecuentes.

El macro recall del 81,24 % indica que el balanceo permite detectar una proporción elevada de los ejemplos pertenecientes a categorías minoritarias, aunque en algunas de ellas se obtiene una precision inferior.


## Rendimiento por producto

Las categorías con mejor F1 son:

| Categoría | F1 |
|---|---:|
| Credit reporting | 92,58 % |
| Mortgage | 84,54 % |
| Student loan | 79,73 % |
| Checking or savings account | 79,53 % |

Las categorías con menor rendimiento son:

| Categoría | F1 |
|---|---:|
| Vehicle loan or lease | 61,26 % |
| Payday, title, personal or advance loan | 51,09 % |

En varias categorías minoritarias, el recall es superior a la precision.

Esto significa que el modelo identifica una parte relevante de los casos reales, pero también asigna estas categorías a reclamaciones que pertenecen a otros productos.


## Threshold de confianza

Además de la categoría predicha, el modelo devuelve la probabilidad máxima como medida operativa de confianza.

El threshold se selecciona utilizando el conjunto de validation.

Se adopta un valor de:

```python
CONFIDENCE_THRESHOLD = 0.70
```

Este valor representa un compromiso entre cobertura y fiabilidad.

### Resultados con threshold

| Métrica | Resultado |
|---|---:|
| Cobertura | 85,77 % |
| Accuracy en predicciones aceptadas | 91,54 % |
| Macro F1 en predicciones aceptadas | 82,67 % |
| Reclamaciones clasificadas como `None` | 8.290 |
| Porcentaje no clasificado automáticamente | 14,23 % |

En un escenario operativo:

- Las predicciones con confianza igual o superior a 0,70 podrían clasificarse automáticamente.
- Las predicciones con menor confianza podrían enviarse a revisión manual o a una segunda etapa de clasificación.

El threshold no mejora el modelo directamente. Permite que el sistema se abstenga en aquellos casos donde la predicción es menos fiable.


## Principales confusiones

Las confusiones más frecuentes aparecen entre productos relacionados semánticamente:

- `Money transfer, virtual currency, or money service` y `Checking or savings account`.
- `Payday, title, personal or advance loan` y `Debt collection or management`.
- `Debt collection or management` y `Credit reporting`.
- `Vehicle loan or lease`, `Credit reporting` y procesos de recobro.
- `Credit or prepaid card` y `Checking or savings account`.

Muchas reclamaciones describen varias fases de un mismo problema financiero, por ejemplo:

- El producto original.
- Un impago.
- El proceso de cobro.
- La aparición de la deuda en el historial crediticio.

Esto puede hacer que más de una categoría resulte semánticamente razonable a partir del texto.


## Análisis de errores

El modelo comete 7.965 errores sobre las 58.249 observaciones del conjunto de test, aproximadamente el 13,7 %.

Los principales patrones observados son:

- Reclamaciones que contienen señales de varias categorías.
- Narrativas ambiguas o incompletas.
- Textos centrados en la consecuencia del problema y no en el producto original.
- Posible ruido en las etiquetas.
- Errores con confianza elevada que el threshold no consigue filtrar.

La revisión cualitativa de los errores es importante para distinguir entre errores claros del modelo y reclamaciones donde varias categorías podrían considerarse válidas.


## Requisitos

El notebook utiliza Python 3 y las siguientes librerías:

```text
pandas
numpy
torch
transformers
datasets
accelerate
scikit-learn
matplotlib
```

Instalación:

```bash
pip install pandas numpy torch transformers datasets accelerate scikit-learn matplotlib
```

Se recomienda ejecutar el notebook con una GPU compatible con CUDA.

Cuando hay una GPU disponible, el entrenamiento utiliza FP16 para reducir el consumo de memoria y acelerar el proceso.


## Ejecución

1. Clonar o descargar el repositorio.

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
```

2. Instalar las dependencias.

```bash
pip install pandas numpy torch transformers datasets accelerate scikit-learn matplotlib
```

3. Crear la carpeta de datos.

```bash
mkdir data
```

4. Descargar el CSV del CFPB.

5. Guardarlo con el siguiente nombre:

```text
data/complaints.csv
```

6. Abrir el notebook:

```text
SEC_BERT.ipynb
```

7. Ejecutar las celdas en orden.

El entrenamiento utilizado en el prototipo tardó aproximadamente 17 minutos sobre GPU. El tiempo puede variar en función del hardware disponible.


## Reproducibilidad

Se utiliza una semilla común para:

- La división en train, validation y test.
- El submuestreo de las categorías.
- El entrenamiento del modelo.

El notebook conserva un único checkpoint y carga al finalizar el modelo con mejor macro F1 de validation.

Aunque las semillas reducen la variabilidad, pueden existir pequeñas diferencias entre ejecuciones debido al hardware, CUDA y algunas operaciones no deterministas.


## Archivos no incluidos

No se incluyen en el repositorio:

- El CSV original.
- Los checkpoints generados durante el entrenamiento.
- El modelo fine-tuned completo.
- Archivos temporales de Jupyter.

El archivo `.gitignore` debería contener:

```gitignore
data/
*.csv
sec_bert_product_classifier/
.ipynb_checkpoints/
__pycache__/
```

## Limitaciones

Las principales limitaciones del prototipo son:

- El entrenamiento se limita a una época.
- Las reclamaciones se truncan a 128 tokens.
- Existe un fuerte desbalanceo entre categorías.
- Algunas narrativas son ambiguas o incluyen varios productos.
- Puede existir ruido en las etiquetas.
- Solo se utiliza el texto de la reclamación.
- Las probabilidades del modelo no se han calibrado explícitamente.
- Las métricas con threshold se calculan únicamente sobre las predicciones aceptadas y deben interpretarse junto con la cobertura.


## Posibles mejoras

Los siguientes pasos serían:

1. Ampliar el número máximo de épocas.
2. Incorporar early stopping.
3. Evaluar longitudes de 256 y 512 tokens.
4. Monitorizar training loss y validation loss por época.
5. Comparar distintas estrategias de balanceo.
6. Evaluar una función de pérdida ponderada.
7. Calibrar las probabilidades.
8. Analizar de forma específica las clases con menor rendimiento.
9. Evaluar el uso de la jerarquía `Product` y `Sub-product`.
10. Definir una segunda etapa para los casos rechazados por el threshold.
11. Monitorizar rendimiento, distribución de categorías y drift en producción.


## Conclusión

El fine-tuning de SEC-BERT proporciona un prototipo sólido para clasificar reclamaciones financieras en nueve familias de producto.

Sobre el conjunto de test completo, el modelo obtiene:

- Una accuracy del 86,33 %.
- Un weighted F1 del 87,20 %.
- Un macro F1 del 73,42 %.

La incorporación de un threshold de confianza de 0,70 permite clasificar automáticamente el 85,77 % de las reclamaciones con:

- Una accuracy del 91,54 %.
- Un macro F1 del 82,67 %.

Los resultados muestran que es posible combinar un nivel elevado de automatización con un mecanismo de abstención para las predicciones menos fiables.

No obstante, las diferencias entre categorías y las limitaciones computacionales indican que todavía existe margen de mejora antes de una posible productivización.


## 

Adicionalmente, el notebook contiene comentarios sobre el planteamiento y conclusiones en los que se explican más extensamente los resultados y metodología aplicada
