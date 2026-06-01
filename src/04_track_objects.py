import numpy as np
import pandas as pd
import os

def calcular_distancia_euclidiana(punto_actual, punto_previo):
    """
    Calcula la distancia geométrica real entre dos coordenadas (x, y)
    para determinar la proximidad de un objeto entre frames consecutivos.
    """
    return np.sqrt((punto_actual[0] - punto_previo[0])**2 + (punto_actual[1] - punto_previo[1])**2)

def main():
    print("==================================================================")
    print(" Copa FutBotMX 2026 - Módulo A2: Algoritmo de Tracking Euclidiano ")
    print("==================================================================")
    
    # Definición de rutas de entrada y salida reales del pipeline
    ruta_entrada_centroides = "results/metrics/raw_centroids.csv"
    ruta_salida_tracking = "results/metrics/tracking_data.csv"
    
    # Validación de existencia de datos previos del pipeline
    if not os.path.exists(ruta_entrada_centroides):
        print(f"\n[!] Error de Pipeline: No se encontró el archivo '{ruta_entrada_centroides}'.")
        print("    Asegúrate de que 'src/03_extract_centroids.py' haya procesado las máscaras primero.")
        return

    # Leer los centroides reales extraídos por OpenCV
    df_raw = pd.read_csv(ruta_entrada_centroides)
    
    if df_raw.empty:
        print("[!] Error: El archivo de centroides base está vacío. No hay objetos que rastrear.")
        return

    print(f"[INFO] Registros crudos cargados: {len(df_raw)}. Iniciando asignación temporal...")

    # Diccionario histórico para guardar las últimas posiciones válidas conocidas
    # Estructura de llave: (tipo, equipo, id_objeto) -> Valor: (x, y)
    historial_objetos = {}
    
    # Límites estrictos del reglamento del torneo (Máximo 2 robots activos por equipo)
    contador_ids_global = {"Equipo_A": 1, "Equipo_B": 1}
    
    datos_rastreados = []
    
    # Procesar de forma estrictamente cronológica frame por frame
    lista_frames = sorted(df_raw['frame'].unique())
    UMBRAL_MAX_MOVIMIENTO = 80  # Máximo de píxeles que un robot puede desplazarse por frame (25 FPS)

    for frame in lista_frames:
        df_frame = df_raw[df_raw['frame'] == frame]
        tiempo_seg = df_frame['tiempo'].iloc[0]
        
        nuevos_objetos_frame = {}
        
        # 1. RASTREO EXCLUSIVO DEL BALÓN (ID fijo = 0)
        balon_frame = df_frame[df_frame['tipo'] == 'balon']
        for _, row in balon_frame.iterrows():
            datos_rastreados.append({
                "frame": frame,
                "tiempo": tiempo_seg,
                "id_objeto": 0,
                "tipo": "balon",
                "equipo": "ninguno",
                "x": int(row['x']),
                "y": int(row['y']),
                "area": int(row['area'])
            })
            nuevos_objetos_frame[("balon", "ninguno", 0)] = (row['x'], row['y'])

        # 2. RASTREO DE ROBOTS POR ESCUADRA (Evita cruces de ID entre rivales)
        robots_frame = df_frame[df_frame['tipo'] == 'robot']
        
        for equipo in ["Equipo_A", "Equipo_B"]:
            robots_escuadro = robots_frame[robots_frame['equipo'] == equipo]
            
            # Filtrar el historial del frame anterior que corresponda únicamente a este equipo
            historial_escuadro = {
                clave: pos for clave, pos in historial_objetos.items()
                if clave[0] == "robot" and clave[1] == equipo
            }
            
            # Iterar sobre las detecciones físicas del frame actual
            for _, row in robots_escuadro.iterrows():
                pos_actual = (row['x'], row['y'])
                id_objeto = None
                
                if historial_escuadro:
                    # Encontrar el centroide del frame anterior con la distancia mínima absoluta
                    clave_mas_cercana = min(
                        historial_escuadro.keys(),
                        key=lambda k: calcular_distancia_euclidiana(pos_actual, historial_escuadro[k])
                    )
                    distancia = calcular_distancia_euclidiana(pos_actual, historial_escuadro[clave_mas_cercana])
                    
                    # Si está dentro del rango dinámico lógico, hereda el ID persistente
                    if distancia < UMBRAL_MAX_MOVIMIENTO:
                        id_objeto = clave_mas_cercana[2]
                        # Remover del registro local para evitar que dos contornos reclamen el mismo ID
                        del historial_escuadro[clave_mas_cercana]
                
                # Si no hubo coincidencia cercana (Frame 1, reingreso tras falta o reinicio de juego)
                if id_objeto is None:
                    id_objeto = contador_ids_global[equipo]
                    contador_ids_global[equipo] += 1
                    
                    # Control de desborde de IDs de acuerdo a los 2 robots por equipo en cancha
                    if contador_ids_global[equipo] > 2:
                        contador_ids_global[equipo] = 1 
                
                # Registrar la posición actual como la última conocida para el siguiente cuadro
                nuevos_objetos_frame[("robot", equipo, id_objeto)] = pos_actual
                
                datos_rastreados.append({
                    "frame": frame,
                    "tiempo": tiempo_seg,
                    "id_objeto": id_objeto,
                    "tipo": "robot",
                    "equipo": equipo,
                    "x": int(row['x']),
                    "y": int(row['y']),
                    "area": int(row['area'])
                })
        
        # Actualizar el mapa de seguimiento con los objetos supervivientes de este frame
        historial_objetos = nuevos_objetos_frame

    # 3. EXPORTACIÓN DEL ENTREGABLE FINAL EN EL FORMATO ACORDADO
    df_final = pd.DataFrame(datos_rastreados)
    df_final = df_final.sort_values(by=["frame", "tipo", "id_objeto"]).reset_index(drop=True)
    
    df_final.to_csv(ruta_salida_tracking, index=False)
    
    print("==================================================================")
    print(f"[OK] Algoritmo ejecutado. Archivo de tracking real generado.")
    print(f"     Destino: {ruta_salida_tracking}")
    print("==================================================================\n")
    print(df_final.head(15).to_string(index=False))

if __name__ == "__main__":
    main()