## 8. Interpretación de Resultados
Tras procesar el archivo `tracking_data.csv` con nuestro script de visualización (`06_generate_visualizations.py`), pudimos sacar las siguientes conclusiones:

* **Mapas de Calor y Actividad:**El mapa de calor (`01_heatmap_global.png`), es evidente que el partido se concentró casi por completo en el medio campo. Prácticamente no hubo juego por las bandas; ambos equipos se fueron al choque directo por el centro, lo que explica la alta cantidad de colisiones y la pelea constante por el balón.

* **Control Territorial (Voronoi):** En cuanto a la distribución de los espacios, los diagramas de Voronoi (`05_voronoi.png`) muestran que el Equipo A estuvo mucho mejor parado. Lograron cubrir más área para recuperar la pelota, mientras que el Equipo B se movió de forma más desordenada y dispersa.

* **Análisis de Pases:** La red de pases (`07_red_pases.png`) nos confirma que las interacciones fueron muy cortas. En lugar de triangular o armar jugadas complejas en equipo, la prioridad de los robots fue casi siempre empujar el balón directo hacia la portería.

## 9. Video Demo
Armamos una demostración de 2 minutos para ver el pipeline completo en acción. En el video se aprecia cómo trabaja la segmentación con SAM 3, el tracking de las trayectorias en tiempo real y cómo se sobrepone el dashboard de métricas directamente al partido.
🔗 **[Ver Video Demo](https://youtube.com/watch?v=simulacion_demo)**

## 10. Reel de Instagram
Para la divulgación mostramos los mejores momentos del desarrollo y del partido, pensado para compartir nuestro trabajo en redes.
🔗 **[Ver Reel en Instagram](https://instagram.com/reel/simulacion_reel_futbotmx)**