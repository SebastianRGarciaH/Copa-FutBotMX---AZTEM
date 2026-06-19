# Weekly Log — AZTEM Copa FutBotMX

---

## Semana 0 — 14 al 22 de mayo

### ✅ Tareas Completadas

**A1 — Segmentación**
-  Crear estructura técnica del repositorio
-  Definir el video base a analizar

**A2 — Tracking**
-  Definir formato del CSV (columnas: frame, time_sec, object_id, object_type, x, y, area, confidence)
-  Investigar centroid tracking y distancia euclidiana
-  Crear README inicial con estructura de secciones
-  Registrar integrantes y objetivos

**A3 — Visualización**
-  Definir las visualizaciones a entregar
-  Proponer diseño visual de la cancha
-  Crear plantilla de matplotlib para graficar
-  Crear checklist de entrega

### ⚠️ Problemas Encontrados

| ID | Problema | Descripción | Solución / Workaround |
|----|----------|-------------|----------------------|
| P-01 | No hay mascaras para pruebas | No se han podido generar las mascaras para primeras pruebas| Simular coordenadas para probar|
| P-02 | Centroides volatiles | Los centroides son muy sensibles a diferencias o cambios | IR estableciendo un sistema geom[etrico para determinar objetos ] 

### 🎯 Próximos Pasos (Semana 1)

-  A1: Extraer frames del video y aplicar SAM a frames individuales
-  A2: Recibir primeras máscaras y probar extracción de centroide
-  A3: Crear función para graficar trayectorias con datos simulados

---

## Semana 1 — 25 al 31 de mayo

### ✅ Tareas Completadas

**A1 — Segmentación**
-  Instalar SAM 2/3 y verificar que corre
-  Extraer frames del video
-  Aplicar SAM a frames individuales
-  Probar segmentación de balón y robots
-  Guardar primeras máscaras en `results/masks/`
-  Iniciar `07_full_pipeline.py` encadenando scripts 01 y 02

**A2 — Tracking**
-  Recibir primeras máscaras de A1
-  Probar extracción de centroide con OpenCV
-  Guardar centroides de balón y robots en CSV preliminar
-  Agregar capturas de primeras segmentaciones al README
-  Verificar que `requirements.txt` funcione

**A3 — Visualización**
-  Trabajar con datos simulados
-  Crear función para graficar trayectorias sobre imagen de cancha
-  Probar seaborn para mapa de calor

### ⚠️ Problemas Encontrados

| ID | Problema | Descripción | Solución / Workaround |
|----|----------|-------------|----------------------|
| P-01 | Complejidad para correr SAM 2/3 | El modelo ocupa muchos recursos | Cambiar a uso mediante Hugging Face |

### 🎯 Próximos Pasos (Semana 2)

-  A1: Procesar un fragmento completo de video y exportar máscaras para A2
-  A2: Implementar tracking con IDs y generar CSV completo
-  A3: Graficar trayectorias reales con el CSV de A2

---

## Semana 2 — 1 al 7 de junio

### ✅ Tareas Completadas

**A1 — Segmentación**
-  Escribir instrucciones iniciales de instalación
-  Procesar un fragmento completo de video
-  Exportar máscaras en formato compatible para A2
-  Documentar errores de segmentación
-  Integrar scripts 03 y 04 al pipeline

**A2 — Tracking**
-  Implementar tracking frame a frame con asignación de IDs
-  Generar `results/metrics/tracking_data.csv` completo
-  Producir video preliminar con puntos y trayectorias
-  Agregar GIF de tracking al README

**A3 — Visualización**
-  Recibir CSV preliminar de A2
-  Graficar trayectorias reales
-  Separar visualmente balón y robots por color
-  Empezar diseño de `08_export_demo_video.py`

### ⚠️ Problemas Encontrados

| ID | Problema | Descripción | Solución / Workaround |
|----|----------|-------------|----------------------|
| P-01 | Centroides sensibles | Centroides muy sensibes al ruido de las máscaras | Establecer mejores parámetros para la detección de objetos |
| P-02 | Rastreo aleatorio | Rastreo tomaba aleatoreamente diferentes objetos como si fueran uno mismo | Incluir distancia eucladiana para poder optimizar las rutas del robot |
| P-03 | Alucinaciones de eventos | En la visualización, el display de eventos (pases, tiros, etc) alucinaba datos| Revisar a detalle el CSV de rastreo, crear otro script únicamente para eventos |
| P-04 | Máscaras sin reconocer pelota | Máscaras muy sensibles a interrupciones por objetos externos y no consideraban la peltoa | Implementación de una detección mediante colores para reconocer la pelota |

### 🎯 Próximos Pasos (Semana 3)

-  A1: Ajustar segmentación y escribir `docs/architecture.md`
-  A2: Limpiar CSV e interpolar puntos faltantes
-  A3: Generar mapa de calor y calcular métricas finales

---

## Semana 3 — 8 al 14 de junio

### ✅ Tareas Completadas

**A1 — Segmentación**
-  Mejorar selección de objetos (mejores puntos o cuadros guía)
-  Apoyar integración con A2
-  Escribir `docs/architecture.md`
-  Integrar scripts 05, 06 y 08 al pipeline

**A2 — Tracking**
-  Limpiar datos (suavizar saltos, interpolar puntos faltantes)
-  Actualizar `docs/weekly_log.md`
-  Entregar CSV limpio a A3
-  Escribir `docs/credits.md`

**A3 — Visualización**
-  Generar mapa de calor final
-  Calcular métricas: distancia recorrida, zona de mayor actividad, posesión aproximada
-  Escribir `docs/limitations.md`

### ⚠️ Problemas Encontrados

| ID | Problema | Descripción | Solución / Workaround |
|----|----------|-------------|----------------------|
| P-01 | Máscaras sensibles a interrumpción | Máscaras muy sensibles a interrupciones por objetos externos (no robots ni pelotas) | Implementación de una delimitación automática y mejor distinción de robots y pelotas |
| P-02 | Rastreo aleatorio | Rastreo tomaba rutas aleatorias cuando habia movimientos bruscos por parte de los objetos | Implementación de un "Algoritm Húngaro" para establecer costo de movimientos|

### 🎯 Próximos Pasos (Semana 4 — ENTREGA FINAL)

-  A1: Verificar que el pipeline corra de inicio a fin con un solo comando
-  A2: Finalizar README completo y `credits.md`
-  A3: Grabar video demo y reel de Instagram

---

## Semana 4 — 15 al 19 de junio ⚠️ ENTREGA FINAL

### ✅ Tareas Completadas

**A1 — Segmentación**
-  Verificar que el pipeline corra de inicio a fin con un solo comando
-  Revisar instrucciones técnicas del README (secciones 6 y 7)
-  Apoyar generación del video final

**A2 — Tracking**
-  Exportar CSV final
-  Documentar limitaciones del tracking en `docs/weekly_log.md`
-  Actualizar sección de resultados del README
-  Finalizar `docs/credits.md`
-  Revisar que el README esté completo y ordenado

**A3 — Visualización**
-  Exportar imágenes finales en alta resolución
-  Escribir interpretación de resultados para sección 8 del README
-  Subir enlaces al README


### 🎯 Checklist Final del Repositorio

-  El repositorio es público
-  El README tiene descripción, integrantes, roles y arquitectura
-  El README tiene instrucciones de instalación y ejecución
-  El README tiene capturas, GIFs o imágenes de resultados
-  El README tiene enlace al video demo (máx 2 minutos)
-  El README tiene enlace al reel de Instagram (mín 30 segundos)
-  Hay `requirements.txt` o `environment.yml` funcionando
-  Hay archivo `LICENSE`
-  Hay `docs/credits.md` y `docs/limitations.md` completos
-  El pipeline corre de inicio a fin sin errores
-  No hay contraseñas, tokens o rutas personales en el código

---

> 📅 **Torneo:** 24 al 26 de junio de 2026 — UPIITA-IPN, Ciudad de México