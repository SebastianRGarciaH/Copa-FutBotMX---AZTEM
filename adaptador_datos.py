"""
================================================================
 adaptador_datos.py
 CONVIERTE EL CSV REAL AL FORMATO QUE ESPERA EL PIPELINE
================================================================
 CSV REAL (entrada): frame, tiempo, id_objeto, tipo, equipo,
                     x, y, area, intervencion_humana
 CSV ESPERADO (salida): frame, tiempo, objeto_id, tipo, x, y
                        + columnas extra preservadas

 Adaptaciones que hace:
   1. Renombra id_objeto -> objeto_id
   2. Convierte tipo "robot" + columna equipo -> robot_aliado/robot_rival
   3. Escala pixeles a centimetros
   4. Preserva 'area' e 'intervencion_humana' para graficas extra
================================================================
"""

import pandas as pd

# -------------------------------------------------------------
# CONFIGURACION - ajustar cuando tu compa confirme
# -------------------------------------------------------------
ARCHIVO_ENTRADA = "tracking_real.csv"
ARCHIVO_SALIDA = "tracking.csv"

# Asignacion equipo -> aliado/rival. Ajusta segun tu equipo:
ROBOTS_ALIADOS = ["Equipo_1", "Equipo_2"]
ROBOTS_RIVALES = ["Equipo_3", "Equipo_4"]

# Conversion de pixeles a cm.
# Los datos ya vienen en cm desde el equipo de vision, asi que = 1.
# Si en el futuro pasan a entregar pixeles, ajusta estos valores.
PX_A_CM_X = 1
PX_A_CM_Y = 1


def adaptar():
    df = pd.read_csv(ARCHIVO_ENTRADA)
    print(f"Leido {ARCHIVO_ENTRADA}: {len(df)} filas")
    print(f"   columnas originales: {list(df.columns)}")

    # 1. Renombrar id_objeto -> objeto_id
    if "id_objeto" in df.columns:
        df = df.rename(columns={"id_objeto": "objeto_id"})

    # 2. Convertir tipo + equipo -> tipo (aliado/rival/balon)
    def clasificar(row):
        if row["tipo"] == "balon":
            return "balon"
        equipo = str(row.get("equipo", ""))
        if equipo in ROBOTS_ALIADOS:
            return "robot_aliado"
        if equipo in ROBOTS_RIVALES:
            return "robot_rival"
        return "robot_aliado"   # fallback

    df["tipo"] = df.apply(clasificar, axis=1)

    # 3. Escalar pixeles a cm
    df["x"] = df["x"] * PX_A_CM_X
    df["y"] = df["y"] * PX_A_CM_Y

    # 4. Preservar columnas extra (area, intervencion_humana)
    columnas_base = ["frame", "tiempo", "objeto_id", "tipo", "x", "y"]
    extras = [c for c in ("equipo", "area", "intervencion_humana")
              if c in df.columns]
    columnas_finales = columnas_base + extras
    df_final = df[columnas_finales]

    df_final.to_csv(ARCHIVO_SALIDA, index=False)

    print(f"\nGuardado {ARCHIVO_SALIDA}: {len(df_final)} filas")
    print(f"   tipos: {df_final['tipo'].value_counts().to_dict()}")
    if "intervencion_humana" in df_final.columns:
        intv = df_final["intervencion_humana"].value_counts().to_dict()
        print(f"   intervencion_humana: {intv}")


if __name__ == "__main__":
    adaptar()