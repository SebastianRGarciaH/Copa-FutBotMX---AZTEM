# 🎬 Guion del Video Demo - Copa FutBotMX

**Equipo:** AZTEM  
**Categoría:** Amateur  
**Duración Objetivo:** 2 Minutos  

---

| Tiempo | Visual (Lo que se ve en pantalla) | Audio (Voz en off / Narración) |
| :--- | :--- | :--- |
| **0:00 - 0:08** | **Presentación Rápida:** Logo del equipo AZTEM y título del proyecto. Transición rápida al problema. | > "Somos AZTEM, de la categoría Amateur. Presentamos nuestra herramienta de análisis de video con IA para la Copa FutBotMX." |
| **0:08 - 0:25** | **El Problema:** Video original del partido reproduciéndose normalmente (puede ser una toma limpia). | > "Durante un partido, es casi imposible analizar a simple vista la posición exacta del balón o el rendimiento y recorrido de cada robot." |
| **0:25 - 0:55** | **Segmentación Multivista:** Pantalla dividida en dos. A un lado el video con **vista aérea** y al otro la **vista lateral** reproduciéndose al mismo tiempo. Ambos muestran las máscaras de segmentación superpuestas sobre los robots y el balón. | > "Para solucionarlo, aplicamos modelos como SAM 2 o 3. Esto nos permite segmentar e identificar automáticamente cada elemento en la cancha, sincronizando múltiples perspectivas, como la vista aérea y lateral simultáneamente." |
| **0:55 - 1:35** | **Tracking y Simulación:** Pantalla dividida. De un lado, el **video original reiniciado** (limpio). Del otro lado, el **GIF de la simulación** mostrando los puntos (centroides) y las líneas de trayectoria, moviéndose en perfecta sincronía con el video. | > "Calculamos los centroides a partir de esas máscaras. Aquí podemos ver el video original sincronizado con nuestra simulación digital. Esto nos permite hacer un seguimiento preciso frame a frame y trazar la trayectoria real de toda la jugada." |
| **1:35 - 1:50** | **Visualizaciones y Resultados:** Pantalla enfocada en el mapa de calor y el dashboard de métricas generado. | > "Toda esta información se traduce en mapas de calor y métricas clave: distancia recorrida por robot, zonas de mayor actividad y el porcentaje de posesión." |
| **1:50 - 2:00** | **Cierre Rápido:** Breve vistazo al repositorio de GitHub haciendo scroll rápido o una placa final de agradecimiento. | > "Este pipeline es la base para el análisis automatizado y futuros detectores de jugadas complejas. ¡Gracias por su atención!" |