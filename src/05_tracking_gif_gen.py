import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def main():
    print("==================================================================")
    print("=== MÓDULO A3: GENERADOR AUTOMÁTICO DE GIF DE RASTREO (TEST) ===")
    print("==================================================================")
    
    ruta_csv = "results/metrics/tracking_data.csv"
    ruta_gif_salida = "results/tracking_visualization.gif"
    
    if not os.path.exists(ruta_csv):
        print(f"[!] No se encontró el archivo: {ruta_csv}. Ejecuta primero el tracker.")
        return

    df = pd.read_csv(ruta_csv)
    if df.empty:
        print("[!] El archivo de tracking está vacío.")
        return

    # Configurar el lienzo de Matplotlib en formato Landscape estricto
    fig, ax = plt.subplots(figsize=(10, 7.5))
    
    frames_unicos = sorted(df['frame'].unique())
    
    # Colores estratégicos por bando
    colores_equipos = {
        "Equipo A": "#007fff",  # Azul eléctrico
        "Equipo B": "#ff2a2a"   # Rojo brillante
    }

    def actualizar_cuadro(num_frame):
        ax.clear()
        
        # 1. Dibujar la cancha virtual (Fondo verde de competencia)
        ax.set_facecolor('#1e7e34')
        ax.set_title(f"Copa FutBotMX - Visualización de Tracking (Frame {num_frame})", fontsize=12, fontweight='bold')
        
        # Límites cartesianos de tu lona en Landscape
        ax.set_xlim(0, 170)
        ax.set_ylim(130, 0) # Origen arriba a la izquierda para simular cámara
        
        # Dibujar líneas básicas de la cancha en blanco
        ax.plot([0, 170, 170, 0, 0], [0, 0, 130, 130, 0], color='white', linewidth=3) # Banda externa
        ax.plot([85, 85], [0, 130], color='white', linewidth=2) # Media cancha
        centro_cancha = plt.Circle((85, 65), 20, color='white', fill=False, linewidth=2)
        ax.add_patch(centro_cancha)
        
        # 2. Filtrar datos del frame actual
        datos_frame = df[df['frame'] == num_frame]
        
        for _, robot in datos_frame.iterrows():
            r_id = robot['id_objeto']
            equipo = robot['equipo']
            x_pos = robot['x']
            y_pos = robot['y']
            intervencion = robot['intervencion_humana']
            
            color = colores_equipos.get(equipo, '#ffcc00')
            
            # Dibujar la "Estela" / Historial de los últimos 10 frames del robot
            historial = df[(df['id_objeto'] == r_id) & (df['frame'] <= num_frame) & (df['frame'] > num_frame - 10)]
            if not historial.empty:
                ax.plot(historial['x'], historial['y'], color=color, alpha=0.3, linewidth=2, linestyle='--')
            
            # Dibujar el robot actual como un chasis circular
            ax.scatter(x_pos, y_pos, color=color, s=400, edgecolors='white', linewidths=2, zorder=5)
            
            # Colocar el ID del robot en el centro del círculo
            ax.text(x_pos, y_pos, str(r_id), color='white', ha='center', va='center', fontweight='bold', fontsize=10, zorder=6)
            
            # Alerta visual corregida: facecolors='none' para hacer el círculo hueco
            if intervencion == "si":
                ax.scatter(x_pos, y_pos, s=700, edgecolors='yellow', facecolors='none', linewidths=3, zorder=4)
                ax.text(x_pos, y_pos - 8, "MANO", color='yellow', ha='center', fontsize=8, fontweight='bold')

        # Cuadro de acotación de equipos
        ax.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='Equipo A (IDs 1-2)', markerfacecolor='#007fff', markersize=12),
            plt.Line2D([0], [0], marker='o', color='w', label='Equipo B (IDs 3-4)', markerfacecolor='#ff2a2a', markersize=12)
        ], loc='upper right')
        
        ax.grid(True, alpha=0.1, color='white')

    print(f"[INFO] Renderizando animación de {len(frames_unicos)} cuadros...")
    ani = animation.FuncAnimation(fig, actualizar_cuadro, frames=frames_unicos, interval=60)
    
    print("[INFO] Guardando archivo GIF en disco duro (esto puede tomar unos segundos)...")
    ani.save(ruta_gif_salida, writer='pillow', fps=1)
    plt.close()
    
    print(f"\n[ÉXITO] ¡GIF de prueba generado de forma espectacular!")
    print(f"[OK] Archivo guardado listo para ver en: {ruta_gif_salida}")
    print("==================================================================")

if __name__ == "__main__":
    main()
