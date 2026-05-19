# AZTEM Copa FutBotMX

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)
![Meta SAM](https://img.shields.io/badge/Model-SAM%202%2F3-black?style=for-the-badge&logo=meta&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Category](https://img.shields.io/badge/Categoría-Amateur-orange?style=for-the-badge)

## 1. Descripción General
Este repositorio contiene un pipeline reproducible de Visión por Computadora e Inteligencia Artificial diseñado para la segmentación, tracking y análisis estadístico de partidos de la Copa FutBotMX 2026. El sistema procesa video original para extraer métricas de juego clave de manera automatizada.

## 2. Integrantes y Roles
* **Sebastián René García Herrero:** Líder Técnico + Módulo de Segmentación (SAM 2/3) e Integración de Pipeline.
* **Juan Pablo Lopez Moreno:** Módulo de Tracking y Estructuración de Datos (CSV).
* **José de Jesús Flores Rojas:** Módulo de Análisis/Visualización y Producción de Video Demo.

## 3. Categoría
Amateur - Análisis de Video con Visión por Computadora.

## 4. Objetivo del Proyecto
Entregar un repositorio público con un pipeline reproducible que use SAM 2/3 para
segmentar robots y balón en videos de FutBotMX, generar tracking básico, visualizaciones del
partido y un video demo claro.

## 5. Arquitectura del Pipeline
*Flujo conceptual general:* Video Original → Extracción de Frames → Segmentación con SAM 2/3 → Cálculo de Centroides → Tracking de Objetos (IDs) → Exportación de CSV → Procesamiento de Métricas y Visualizaciones → Video Demo Final.

## 6. Instalación
*Aquí se incluirán las instrucciones detalladas para clonar el repositorio e instalar las dependencias necesarias mediante `requirements.txt` o `environment.yml`.*

## 7. Cómo Ejecutar
*Aquí se detallarán los comandos para correr de manera independiente cada script de la carpeta `src/` y el comando único para ejecutar el pipeline completo (`src/07_full_pipeline.py`).*

## 8. Resultados y Formato de Datos

### Estructura Definitiva del CSV (`results/metrics/tracking_data.csv`)
Los datos de posicionamiento y telemetría extraídos frame a frame se estructuran bajo el siguiente formato contractual de datos:

| Columna | Tipo de Dato | Ejemplo | Descripción / Regla de Validación |
| :--- | :--- | :--- | :--- |
| `frame` | `Int` | `105` | Número secuencial del cuadro del video. |
| `segundo` | `Float` | `3.50` | Tiempo transcurrido en segundos desde el inicio del video. |
| `id_objeto` | `Int` | `2` | ID único asignado de forma persistente a un objeto por el algoritmo de tracking. |
| `tipo_objeto` | `String` | `"robot"` | Tipo de objeto detectado (`"robot"` o `"ball"`). |
| `equipo` | `String` | `"UPIITA_Bots"` | Nombre oficial del equipo al que pertenece el robot (`"none"` para el balón). |
| `x` | `Int` / `Float` | `640` | Coordenada en el eje X del centroide del objeto. |
| `y` | `Int` / `Float` | `480` | Coordenada en el eje Y del centroide del objeto. |
| `area` | `Int` | `1250` | Cantidad de píxeles que conforman la máscara binaria del objeto. |
| `confianza` | `Float` | `0.95` | Nivel de confianza de la detección/tracking (`0.0` a `1.0`). |

### Ejemplo de Vista Previa del CSV
```csv
frame,segundo,id_objeto,tipo_objeto,equipo,x,y,area,confianza
105,3.50,0,ball,none,520,315,180,0.99
105,3.50,1,robot,UPIITA_Bots,450,300,1200,0.94
105,3.50,2,robot,MechaMasters,580,330,1150,0.91
```
## 13. Licencia
Este proyecto está bajo la [Licencia MIT](LICENSE). Puedes revisar el archivo para más detalles.