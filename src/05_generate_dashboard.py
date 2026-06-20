"""
================================================================
 05_generate_dashboard.py — Copa FutBotMX · AZTEM
 Dashboard Interactivo (Versión Web Ligera)
================================================================
"""

import os
import pandas as pd
import numpy as np
import webbrowser

# ================================================================
# RUTAS DEL PROYECTO
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROYECTO = os.path.dirname(BASE_DIR)

CARPETA_METRICAS = os.path.join(RAIZ_PROYECTO, "results", "metrics")
CARPETA_VISUALIZACIONES = os.path.join(RAIZ_PROYECTO, "results", "visualizations")

ARCHIVO_TRACKING = os.path.join(CARPETA_METRICAS, "tracking_data.csv")
ARCHIVO_EVENTOS = os.path.join(CARPETA_METRICAS, "game_events.csv")

os.makedirs(CARPETA_VISUALIZACIONES, exist_ok=True)
ARCHIVO_HTML_SALIDA = os.path.join(CARPETA_VISUALIZACIONES, "08_dashboard.html")

EQUIPO_A = "Equipo A"
EQUIPO_B = "Equipo B"

# ================================================================
# LÓGICA DE CARGA DE DATOS
# ================================================================
def cargar_tracking(ruta):
    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip().str.lower()
    if "id_objeto" in df.columns and "objeto_id" not in df.columns:
        df.rename(columns={"id_objeto": "objeto_id"}, inplace=True)
    df["equipo"] = df["equipo"].astype(str).str.strip()
    df["tipo"] = df["tipo"].astype(str).str.strip().str.lower()

    def _clasificar(fila):
        if fila["tipo"] == "balon": return "balon"
        if fila["equipo"] == EQUIPO_A: return "robot_aliado"
        if fila["equipo"] == EQUIPO_B: return "robot_rival"
        return "desconocido"

    df["tipo_ext"] = df.apply(_clasificar, axis=1)
    df = df.dropna(subset=["x", "y"])
    return df

def cargar_eventos(ruta):
    if not os.path.exists(ruta):
        return pd.DataFrame(columns=["evento", "equipo"])
    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip().str.lower()
    df["evento"] = df["evento"].str.strip()
    df["equipo"] = df["equipo"].str.strip()
    return df

# ================================================================
# CÁLCULO DE MÉTRICAS
# ================================================================
def calcular_metricas():
    if not os.path.exists(ARCHIVO_TRACKING):
        print(f"[ERROR] No se encontró {ARCHIVO_TRACKING}")
        return {}

    tracking = cargar_tracking(ARCHIVO_TRACKING)
    eventos = cargar_eventos(ARCHIVO_EVENTOS)

    aliados = tracking[tracking["tipo_ext"] == "robot_aliado"]
    rivales = tracking[tracking["tipo_ext"] == "robot_rival"]

    def dist_total(df):
        total = 0.0
        for _, g in df.groupby("objeto_id"):
            g = g.sort_values("frame")
            total += np.hypot(g["x"].diff(), g["y"].diff()).sum()
        return total

    controles = eventos[eventos["evento"] == "Control de Balon"]
    pct_a = 0
    if len(controles) > 0:
        pct_a = round((controles["equipo"].str.strip() == EQUIPO_A).sum() / len(controles) * 100)

    return {
        "Robots Eq. A": aliados["objeto_id"].nunique(),
        "Robots Eq. B": rivales["objeto_id"].nunique(),
        "Frames Analizados": tracking["frame"].nunique(),
        "Dist. A (cm)": f"{dist_total(aliados):.0f}",
        "Dist. B (cm)": f"{dist_total(rivales):.0f}",
        "Tiros a Gol": int((eventos["evento"] == "Tiro a Gol / Despeje").sum()),
        "Controles Balón": len(controles),
        "Posesión A": f"{pct_a}%",
        "Intervenciones": int((tracking.get("intervencion_humana", pd.Series(dtype=str)).astype(str).str.lower() == "si").sum())
    }

# ================================================================
# GENERACIÓN DE HTML (Plantilla Simétrica)
# ================================================================
def generar_html(metricas):
    stats_html = ""
    for titulo, valor in metricas.items():
        stats_html += f"""
        <div class="stat-item">
            <div class="stat-value">{valor}</div>
            <div class="stat-label">{titulo}</div>
        </div>
        """

    # (archivo, nombre, descripcion)  -> la descripción se muestra de forma dinámica
    vistas = [
        ("01_heatmap.png", "Mapa de Calor", "Zonas de mayor actividad y concentración de los robots a lo largo del partido."),
        ("02_trayectorias.png", "Trayectorias", "Recorrido completo de cada robot durante la secuencia analizada."),
        ("03_pases.png", "Mapa de Pases", "Conexiones de pase entre robots aliados y la direccionalidad del juego."),
        ("04_tiros.png", "Mapa de Tiros", "Ubicación y origen de los tiros a gol y despejes registrados."),
        ("05_voronoi.png", "Control Voronoi", "Dominio territorial: área de la cancha controlada por cada equipo."),
        ("06_posesion.png", "Posesión por Zonas", "Distribución de la posesión del balón segmentada por zonas de la cancha."),
        ("07_red_pases.png", "Red de Pases", "Estructura de la red de pases y los nodos (robots) más conectados.")
    ]

    # Opciones del menú desplegable
    opciones_html = ""
    for i, (archivo, nombre, _desc) in enumerate(vistas):
        sel = "selected" if i == 0 else ""
        opciones_html += f'<option value="{archivo}" {sel}>{nombre}</option>\n'

    # Diccionario JS con las descripciones de cada vista
    descripciones_js = ", ".join(
        [f'"{archivo}": "{desc}"' for archivo, _nombre, desc in vistas]
    )

    # Datos iniciales (primera vista)
    img_inicial = vistas[0][0]
    desc_inicial = vistas[0][2]

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resumen del Partido · AZTEM</title>
    <style>
        :root {{
            --fondo: #f0f4f8;
            --texto: #102a43;
            --texto-tenue: #627d98;
            --acento: #005bb5;
            --acento-hover: #004488;
            --borde: #d9e2ec;
            --blanco: #ffffff;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: var(--fondo);
            color: var(--texto);
            padding: 25px 40px;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}

        header {{
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--borde);
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        header h1 {{ font-size: 28px; font-weight: 800; color: var(--texto); letter-spacing: -0.5px; }}

        .grid-container {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            column-gap: 25px;
            row-gap: 14px;
            width: 100%;
            max-width: 1800px;
            margin: 0 auto;
        }}

        .panel {{
            background: var(--blanco);
            border: 1px solid var(--borde);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 12px rgba(16, 42, 67, 0.04);
            display: flex;
            flex-direction: column;
        }}
        
        .panel-full {{
            grid-column: 1 / -1;
            padding: 22px 24px;
        }}
        
        .panel-title {{
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--texto-tenue);
            margin-bottom: 4px;
        }}

        /* Subtítulo descriptivo de cada apartado (en negro para que resalte) */
        .panel-subtitle {{
            font-size: 12px;
            color: var(--texto);
            line-height: 1.45;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid var(--fondo);
            min-height: 30px;
        }}

        /* ESTE CONTENEDOR ES LA CLAVE DE LA SIMETRÍA */
        .media-wrapper {{
            flex-grow: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--fondo);
            border-radius: 8px;
            padding: 15px;
            overflow: hidden;
            height: 380px; /* ALTURA FORZADA ESTRICTA */
            width: 100%;
        }}

        video, img.media-content {{
            width: 100%;
            height: 100%;
            object-fit: contain; /* Encaja sin distorsionar */
            object-position: center; /* Mantiene la imagen perfectamente al centro */
            border-radius: 4px;
        }}

        /* ===== MENÚ DESPLEGABLE COMPACTO ===== */
        .selector-wrapper {{
            margin-bottom: 12px;
        }}
        .vista-select {{
            width: 100%;
            background-color: var(--fondo);
            border: 1px solid var(--borde);
            color: var(--texto);
            padding: 11px 40px 11px 14px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            appearance: none;
            -webkit-appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%23005bb5' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 14px center;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}
        .vista-select:hover {{
            border-color: var(--acento);
        }}
        .vista-select:focus {{
            outline: none;
            border-color: var(--acento);
            box-shadow: 0 0 0 3px rgba(0, 91, 181, 0.15);
        }}

        /* Descripción dinámica de la vista seleccionada (en negro para que resalte) */
        .vista-desc {{
            font-size: 12px;
            color: var(--texto);
            line-height: 1.5;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid var(--fondo);
            min-height: 34px;
        }}

        /* ===== RESUMEN EN REJILLA DE TARJETAS ===== */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            width: 100%;
        }}
        .stat-item {{
            background: var(--fondo);
            border: 1px solid var(--borde);
            border-radius: 10px;
            padding: 20px 14px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: 800;
            color: var(--acento);
            line-height: 1.1;
            margin-bottom: 10px;
            letter-spacing: -0.5px;
            word-break: break-word;
        }}
        .stat-label {{
            font-size: 11px;
            font-weight: 700;
            color: var(--texto-tenue);
            text-transform: uppercase;
            letter-spacing: 0.6px;
            line-height: 1.3;
        }}

        @media (max-width: 1200px) {{
            .grid-container {{
                grid-template-columns: 1fr 1fr;
            }}
            .panel-full {{
                grid-column: 1 / -1;
            }}
        }}
        @media (max-width: 768px) {{
            .grid-container {{
                grid-template-columns: 1fr;
            }}
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>

    <header>
        <div>
            <h1>Dashboard Analítico</h1>
        </div>
    </header>

    <div class="grid-container">
        
        <div class="panel">
            <div class="panel-title">Video Original</div>
            <div class="panel-subtitle">Grabación cruda del partido, sin procesar, tal como fue capturada.</div>
            <div class="media-wrapper">
                <video controls>
                    <source src="../../data/partido.mp4" type="video/mp4">
                    Tu navegador no soporta video.
                </video>
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">Rastreo Analizado</div>
            <div class="panel-subtitle">Detección y seguimiento de robots y balón cuadro por cuadro.</div>
            <div class="media-wrapper">
                <img class="media-content" src="tracking_visualization.gif" alt="GIF de tracking">
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">Métricas Espaciales</div>
            <div class="selector-wrapper">
                <select id="vista-selector" class="vista-select" onchange="cambiarVista(this.value)">
                    {opciones_html}
                </select>
            </div>
            <div class="vista-desc" id="vista-desc">{desc_inicial}</div>
            <div class="media-wrapper" style="background: transparent; padding: 0;">
                <img class="media-content" id="vista-img" src="{img_inicial}" alt="Vista seleccionada">
            </div>
        </div>

        <div class="panel panel-full">
            <div class="panel-title">Resumen del Partido</div>
            <div class="panel-subtitle" style="min-height: auto;">Indicadores clave del análisis: movimiento, posesión y eventos de juego.</div>
            <div class="stats-grid">
                {stats_html}
            </div>
        </div>

    </div>

    <script>
        var descripciones = {{ {descripciones_js} }};

        function cambiarVista(src) {{
            document.getElementById("vista-img").src = src;
            var desc = descripciones[src] || "";
            document.getElementById("vista-desc").textContent = desc;
        }}
    </script>
</body>
</html>
"""
    with open(ARCHIVO_HTML_SALIDA, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return ARCHIVO_HTML_SALIDA

# ================================================================
# EJECUCIÓN 
# ================================================================
def main():
    print("=" * 64)
    print("=== Copa FutBotMX · AZTEM — Dashboard ===")
    print("=" * 64)
    metricas = calcular_metricas()
    
    if metricas:
        html_path = generar_html(metricas)
        ruta_absoluta = f"file://{os.path.abspath(html_path)}"
        
        print(f"\n[OK] Datos procesados. Dashboard guardado en:\n  -> {html_path}")
        print("[INFO] Abriendo el dashboard en tu navegador predeterminado automáticamente...")
        
        # Abre directamente en el navegador sin intentar cargar librerías de UI de escritorio
        webbrowser.open(ruta_absoluta)

if __name__ == "__main__":
    main()