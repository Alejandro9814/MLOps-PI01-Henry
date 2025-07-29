
# 🎬 Sistema de Recomendación de Películas

Este proyecto es una **API RESTful** desarrollada con **FastAPI** que permite realizar recomendaciones personalizadas de películas, consultar información sobre títulos específicos, obtener datos estadísticos del catálogo y mucho más.

🔗 **Deploy en Render:**  
[Render](https://mlops-pi01-henry.onrender.com)

---

## 🚀 Tecnologías utilizadas

- **Lenguaje:** Python 3.10  
- **Framework:** FastAPI  
- **ML/DS stack:** pandas, numpy, ast, scikit-learn  
- **Base de datos:** pandas + archivos `.parquet` locales  
- **Deploy:** Render  
- **Servidor ASGI:** Uvicorn  

---

## 📌 Endpoints disponibles

| Endpoint                            | Descripción                                                         |
|-------------------------------------|----------------------------------------------------------------------|
| `/`                                 | Mensaje de bienvenida                                               |
| `/cantidad_estrenos_mes?mes=`       | Películas estrenadas en un mes específico                           |
| `/cantidad_filmaciones_dia?dia=`    | Películas estrenadas en un día específico                           |
| `/score_titulo?titulo_de_la_filmación=` | Año y popularidad de una película                              |
| `/votos_titulo?titulo_de_la_filmación=` | Votos totales y promedio si supera las 2000 valoraciones      |
| `/get_actor?nombre_actor=`          | Métricas de participación de un actor                               |
| `/get_director?nombre_director=`    | Películas dirigidas por un director y métricas asociadas            |
| `/get_recomendacion?titulo=`        | Recomendaciones basadas en similitud de contenido                   |

---

## 🧱 Arquitectura del Proyecto

```
├── DataSet/                     # Archivos .parquet y matriz de similitud
│   ├── Cast_Movies.parquet
│   ├── cosine_similarity_matrix.pkl
│   └── ...
├── ETL/                         # Jupyter Notebooks y scripts de procesamiento
│   ├── Transformación_datos_ML.ipynb
│   └── ...
├── Deploy/                      # Carpeta principal para FastAPI
│   └── main.py
├── requirements.txt             # Librerías necesarias
├── README.md                    # Documentación del proyecto
├── ER_Movie_Model.png           # Modelo Entidad-Relación
└── .gitignore
```

---

## ⚙️ Cómo Correrlo Localmente

Cloná el repositorio:  
`git clone https://github.com/Alejandro9814/MLOps-PI01-Henry.git`  
`cd MLOps-PI01-Henry`

Instalá las dependencias:  
`pip install -r requirements.txt`

Ejecutá el servidor:  
`uvicorn main:app --reload`

Accedé a la documentación en:

- Swagger UI: http://localhost:8000/docs  
- ReDoc: http://localhost:8000/redoc

---

## 📊 Dataset

El dataset fue provisto por Henry para fines académicos y contiene información detallada sobre películas, directores, actores, géneros y compañías de producción.

[Diccionario de Datos](https://docs.google.com/spreadsheets/d/1QkHH5er-74Bpk122tJxy_0D49pJMIwKLurByOfmxzho/edit?gid=0#gid=0)

---

## 🔄 Proceso de Transformación de los Datos

Para construir el sistema de recomendación y los distintos endpoints de consulta, se realizó un proceso de transformación profunda sobre el dataset original, que contenía múltiples campos anidados y estructuras no normalizadas. El objetivo fue optimizar los datos para su análisis, permitir consultas eficientes y generar una base sólida para el sistema de recomendaciones.

### Desanidamiento y Normalización

- Se desanidaron columnas como:
  - **Actores y Directores**: originalmente contenían estructuras JSON con múltiples registros por película. Se transformaron en tablas separadas vinculadas por `id`.
  - **Géneros y Colecciones**: se extrajeron y estructuraron en columnas independientes para facilitar filtrado, análisis y modelado.
- Se manejaron valores nulos, se estandarizaron formatos (por ejemplo, fechas y nombres) y se eliminaron registros inconsistentes.

### Arquitectura de Datos y Modelo Relacional

- Se diseñó una arquitectura **tipo estrella**, donde:
  - La tabla principal contiene la información de las películas (tabla de hechos).
  - Tablas auxiliares (dimensiones) contienen información relacionada como actores, directores, productoras y géneros.
- Esta estructura relacional permite un acceso más ordenado, reutilizable y eficiente para la lógica de negocio implementada en los endpoints.

### Optimización con Formato Parquet

- Todos los archivos procesados fueron convertidos a **formato Parquet**, lo cual:
  - Reduce significativamente el tamaño del almacenamiento.
  - Mejora los tiempos de lectura en memoria.
  - Facilita la integración con bibliotecas como `pandas` para consultas rápidas en entorno productivo.

### Preparación para Machine Learning

- Se generaron variables a partir de metadatos como géneros, keywords y compañías.
- Se aplicó **vectorización TF-IDF** sobre los textos relevantes.
- Se calculó una matriz de **similitud del coseno** entre películas, almacenada como archivo `.pkl`, utilizada en el endpoint de recomendaciones.

---

## 🧠 Sistema de Recomendación

El sistema de recomendación implementado se basa en la similitud de contenido mediante:

- **TF-IDF vectorization** de los metadatos de películas (como géneros, palabras clave, etc.).
- **Cálculo de similitud del coseno** para encontrar películas similares.
- **Uso de una matriz de similitud precalculada** almacenada como `cosine_similarity_matrix.pkl`.

El usuario puede obtener recomendaciones llamando al endpoint:  
`/get_recomendacion?titulo=Nombre de la Película`

---

## 👨‍💻 Autor

Alejandro Oldra
