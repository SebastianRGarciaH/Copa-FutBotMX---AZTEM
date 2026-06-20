"""
run_pipeline.py — Corredor maestro del pipeline Copa FutBotMX (Equipo AZTEM)

Por qué existe este archivo: el orden NUMÉRICO de los nombres de los
scripts (00, 01, 02... 08) NO coincide con el orden REAL en el que deben
ejecutarse según sus dependencias de datos. En particular:

  - 07_detect_events.py debe correr ANTES que 06_generate_visualizations.py
    (varias vistas de 06 -- pases, tiros, posesión, red de pases -- leen
    el CSV de eventos que genera 07).
  - 05_generate_dashboard.py debe correr DESPUÉS de 06 y 07 (consume las
    imágenes que genera 06 y el CSV de eventos de 07).

Si alguien corre los scripts sueltos siguiendo el número del nombre de
archivo, el dashboard y varias visualizaciones saldrán vacías o
incompletas. Este script corre todo en el orden de dependencias real.

Colocar este archivo en la RAÍZ del proyecto (al mismo nivel que las
carpetas data/, results/ y src/).

Uso:
    python run_pipeline.py                # corre todo el pipeline
    python run_pipeline.py --desde 03      # reanuda desde un paso específico
    python run_pipeline.py --solo 06       # corre un único paso
"""
import os
import subprocess
import sys
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(ROOT_DIR, "src")

# (código, archivo, es_crítico)
# es_crítico=True  -> si falla, se detiene todo el pipeline (algo aguas
#                      abajo seguro también va a fallar).
# es_crítico=False -> si falla, se avisa y se continúa (ej. el paso 08
#                      depende de la vista lateral, que puede no existir
#                      aún -- no tiene sentido tumbar todo el pipeline
#                      por el video demo final).
PASOS = [
    ("01", "01_extract_frames.py",          True),
    ("00", "00_preprocess.py",              True),
    ("02", "02_segment_with_sam.py",        True),
    ("03", "03_extract_centroids.py",       True),
    ("04", "04_track_objects.py",           True),
    ("07", "07_detect_events.py",           True),
    ("06", "06_generate_visualizations.py", True),
    ("05", "05_generate_dashboard.py",      True),
    ("08", "08_export_demo_video.py",       False),
]


def correr_paso(codigo, archivo, critico, env):
    ruta = os.path.join(SRC_DIR, archivo)
    print("\n" + "=" * 70)
    print(f"PASO {codigo}: {archivo}")
    print("=" * 70)

    if not os.path.exists(ruta):
        print(f"[ERROR] No se encontró {ruta}")
        if critico:
            sys.exit(1)
        return 1

    resultado = subprocess.run([sys.executable, ruta], cwd=ROOT_DIR, env=env)

    if resultado.returncode != 0:
        print(f"\n[ERROR] {archivo} terminó con código {resultado.returncode}.")
        if critico:
            print("Este paso es crítico para el resto del pipeline. Deteniendo.")
            sys.exit(resultado.returncode)
        else:
            print("Este paso NO es crítico (ej. depende de la vista lateral, "
                  "aún sin resolver) -- se continúa con el resto del pipeline.")
    return resultado.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desde", default=None,
                         help="Código del paso desde el cual reanudar (ej. 03)")
    parser.add_argument("--solo", default=None,
                         help="Código de un único paso a correr (ej. 06)")
    args = parser.parse_args()

    env = os.environ.copy()
    env["AUTO_CONFIRM"] = "1"  # evita que 00_preprocess.py se quede esperando input()

    if args.solo:
        paso = next((p for p in PASOS if p[0] == args.solo), None)
        if paso is None:
            print(f"[ERROR] No existe el paso '{args.solo}'.")
            sys.exit(1)
        correr_paso(*paso, env=env)
        return

    empezar = args.desde is None
    for codigo, archivo, critico in PASOS:
        if not empezar:
            if codigo == args.desde:
                empezar = True
            else:
                continue
        correr_paso(codigo, archivo, critico, env)

    print("\n" + "=" * 70)
    print("[LISTO] Pipeline completo.")
    print("  Máscaras:        results/masks/")
    print("  Métricas:        results/metrics/")
    print("  Visualizaciones: results/visualizations/")
    print("  Dashboard:       results/visualizations/08_dashboard.html")
    print("  Video demo:      results/demo_final.mp4 (si el paso 08 fue exitoso)")
    print("=" * 70)


if __name__ == "__main__":
    main()
