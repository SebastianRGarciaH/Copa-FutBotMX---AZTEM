import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS DE DISEÑO
# -------------------------------------------------------------
RUTA_CSV_TRACKING = "results/metrics/tracking_data.csv"
RUTA_GIF_SALIDA = "results/tracking_visualization.gif"
VENTANA_ESTELA = 10     # Cuántos frames atrás dibuja la línea de rastro
FPS = 15                 # Cuadros por segundo de la animación

# Paleta tipográfica y de color institucional de AZTEM
PALETA_COLORES = {
    "Equipo A": "#0066cc",    # Azul Aliado
    "Equipo B": "#e8553a",    # Rojo Rival
    "balon": "#ff9f43",       # Naranja Balón
    "aviso": "#ffd200"        # Amarillo Alerta (Mano)
}

def dibujar_cancha_nativa(ax):
    """Dibuja un lienzo blanco con marcas reglamentarias de 170x130 cm."""
    ax.set_facecolor('white')
    
    # 1. Marco exterior de la cancha (170 cm de largo x 130 cm de ancho)
    ax.plot([0, 170, 170, 0, 0], [0, 0, 130, 130, 0], color='#2c3e50', linewidth=2.5, zorder=1)
    
    # 2. Línea de medio campo
    ax.plot([85, 85], [0, 130], color='#2c3e50', linewidth=2, zorder=1)
    
    # 3. Círculo Central (Radio de 20 cm)
    circulo_central = plt.Circle((85, 65), 20, color='#2c3e50', fill=False, linewidth=2, zorder=1)
    ax.add_patch(circulo_central)
    
    # 4. Punto Central de Saque
    ax.scatter(85, 65, color='#2c3e50', s=25, zorder=2)
    
    # 5. Áreas / Porterías de referencia (Bordes de lona)
    # Portería Izquierda (Equipo A)
    ax.plot([0, 12, 12, 0], [40, 40, 90, 90], color='#2c3e50', linewidth=1.5, zorder=1)
    # Portería Derecha (Equipo B)
    ax.plot([170, 158, 158, 170], [40, 40, 90, 90], color='#2c3e50', linewidth=1.5, zorder=1)
    
    # Ajustes físicos de límites de visualización
    ax.set_xlim(-10, 180)
    ax.set_ylim(-10, 140)
    
    # Inversión obligatoria del eje Y para emparejar la perspectiva de la cámara (Matriz OpenCV)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.3, color='#bdc3c7')

def main():
    print("==================================================================")
    print("=== MÓDULO A3: GENERADOR DE RENDERIZADO VISUAL AUTÓNOMO =========")
    print("==================================================================")

    if not os.path.exists(RUTA_CSV_TRACKING):
        print(f"[!] No se encontró el archivo de datos: {RUTA_CSV_TRACKING}")
        return

    # Lectura directa del CSV sin adaptadores intermedios
    df = pd.read_csv(RUTA_CSV_TRACKING)
    if df.empty:
        print("[!] El archivo de tracking está vacío.")
        return

    frames_unicos = sorted(df["frame"].unique())
    fig, ax = plt.subplots(figsize=(11, 8))

    def actualizar_cuadro(num_frame):
        ax.clear()

        # 1. Dibujar la estructura espacial de la cancha blanca
        dibujar_cancha_nativa(ax)

        # Configuración de cabeceras de visualización dinámicas
        datos_frame = df[df["frame"] == num_frame]
        t_val = datos_frame["tiempo"]
        t_txt = f"t = {t_val.iloc[0]:.2f} s" if len(t_val) else "0.00 s"
        
        ax.set_title(f"Copa FutBotMX - Rastreo Cinemático Real\nFrame {num_frame}  ·  {t_txt}", 
                     fontsize=12, fontweight='bold', color='#2c3e50', pad=10)

        # 2. Renderizado capa por capa de los objetos activos
        for _, fila in datos_frame.iterrows():
            r_id = int(fila["id_objeto"])
            tipo = fila["tipo"]
            equipo = fila["equipo"]
            x_pos, y_pos = fila["x"], fila["y"]
            
            # Asignación de color según bando táctico
            color_firma = PALETA_COLORES.get(equipo, PALETA_COLORES["aviso"])

            # 📋 RENDER DE TRAYECTORIA (Estela histórica hacia atrás)
            historial = df[(df["id_objeto"] == r_id) &
                           (df["frame"] <= num_frame) &
                           (df["frame"] > num_frame - VENTANA_ESTELA)
                          ].sort_values("frame")
            
            if len(historial) > 1:
                ax.plot(historial["x"], historial["y"], color=color_firma,
                        alpha=0.35, linewidth=2, linestyle="--", zorder=3)

            # 📋 CONSTRUCCIÓN GEOMÉTRICA DE CHASSIS / BALÓN
            if tipo == "balon":
                # Balón oficial: Pequeño, denso y sin texto interno
                ax.scatter(x_pos, y_pos, color=color_firma, s=120,
                           edgecolors="white", linewidths=1.5, zorder=5)
            else:
                # Chasis de Robot: Círculo robusto con ID de hardware incrustado
                ax.scatter(x_pos, y_pos, color=color_firma, s=380,
                           edgecolors="white", linewidths=2, zorder=4)
                
                ax.text(x_pos, y_pos, str(r_id), color="white", ha="center",
                        va="center", fontweight="bold", fontsize=10, zorder=5)

                # Alerta reglamentaria de arrastre manual ("MANO")
                if str(fila.get("intervencion_humana", "")).lower() == "si":
                    ax.scatter(x_pos, y_pos, s=760, edgecolors=PALETA_COLORES["aviso"],
                               facecolors="none", linewidths=2.5, zorder=3)
                    ax.text(x_pos, y_pos + 7, "MANO", color=PALETA_COLORES["aviso"],
                            ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=5)

        # 3. Leyenda institucional integrada en el marco superior derecho
        ax.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='none', label='Robots Aliados', markerfacecolor=PALETA_COLORES["Equipo A"], markersize=10),
            plt.Line2D([0], [0], marker='o', color='none', label='Robots Rivales', markerfacecolor=PALETA_COLORES["Equipo B"], markersize=10),
            plt.Line2D([0], [0], marker='o', color='none', label='Balón Oficial', markerfacecolor=PALETA_COLORES["balon"], markersize=8)
        ], loc="upper right", framealpha=0.9)

    print(f"[INFO] Renderizando secuencialmente {len(frames_unicos)} cuadros...")
    ani = animation.FuncAnimation(fig, actualizar_cuadro, frames=frames_unicos, interval=1000 / FPS)

    print("[INFO] Guardando animación en disco (Pillow Engine)...")
    os.makedirs(os.path.dirname(RUTA_GIF_SALIDA), exist_ok=True)
    ani.save(RUTA_GIF_SALIDA, writer="pillow", fps=FPS, dpi=110)
    plt.close()

    print(f"\n[ÉXITO] Archivo GIF consolidado en: {RUTA_GIF_SALIDA}")
    print("==================================================================")

if __name__ == "__main__":
    main()