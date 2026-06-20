# Limitaciones y Trabajo Futuro

## 1. Oclusiones y Pérdida de Seguimiento (Tracking)
Durante los partidos, es común que los robots colisionen o se agrupen alrededor del balón. Cuando ocurre una oclusión severa, el modelo pierde temporalmente la segmentación del objeto. Aunque el script de tracking intenta mantener el ID, las desapariciones prolongadas generan pequeños "saltos" en las trayectorias y pueden subestimar las métricas de distancia total recorrida.

## 2. Perspectiva y Distorsión de la Cámara
Las visualizaciones generadas en `06_generate_visualizations.py` (como el diagrama de Voronoi y los mapas de calor) mapean las coordenadas `(x, y)` en píxeles directamente sobre la plantilla 2D de la cancha. Dado que no se aplicó una matriz de homografía para corregir la perspectiva de la cámara, las distancias calculadas presentan un ligero margen de error espacial, especialmente en los extremos del campo.

## 3. Detección Heurística de Eventos
En el archivo `game_events.csv`, eventos complejos como "Posesión" o "Pase" se infirieron utilizando umbrales de distancia euclidiana entre los centroides de los robots y el balón. Esta heurística es rápida, pero puede registrar falsos positivos si un robot simplemente cruza cerca del balón sin ejercer un control o impacto real sobre el mismo.

## Trabajo Futuro
* Integrar un Filtro de Kalman o DeepSORT para predecir trayectorias durante las oclusiones y mantener la consistencia de los IDs.
* Implementar una calibración de cámara para transformar las coordenadas de píxeles a metros reales.