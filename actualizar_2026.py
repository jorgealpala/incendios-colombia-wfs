"""
============================================================================
ACTUALIZACION DIARIA 2026 - Dashboard de incendios Colombia (OroraTech WFS)
============================================================================
Este script se corre A DIARIO durante la temporada del Fenomeno del Nino.
A diferencia de 'preprocesar.py' (que reprocesa TODO desde cero), este solo
se ocupa del anio 2026 y lo fusiona con los anios cerrados ya procesados,
para que la actualizacion sea rapida y el push a GitHub sencillo.

QUE HACE:
  1. Reutiliza las funciones de preprocesar.py (NO duplica logica).
  2. Lee SOLO los geojson de data/2026/, que pueden venir de dos formas:
       (a) Por departamento, renombrados a mano: 'antioquia.geojson',
           'bogota.geojson', etc. (igual que en anios cerrados).
       (b) Combinados diarios: un solo archivo con TODOS los departamentos,
           ej. 'wfs-area-export_2026-05-29-2026-05-29.geojson'.
     No necesitas distinguirlos: el spatial join clasifica ambos, y la regla
     de Bogota maneja los dos casos (por nombre de archivo o por municipio).
  3. Hace el spatial join contra los limites DIVIPOLA (misma logica que siempre).
  4. DEDUPLICA por 'id': si re-descargaste un dia que ya tenias, no se duplica.
  5. Fusiona 2026 con los anios cerrados (lee data/processed/incendios.parquet
     si existe, o el CSV) y reescribe la tabla principal combinada.
  6. NO vuelve a simplificar los limites (esos no cambian dia a dia). Si algun
     dia quieres regenerarlos, corre 'preprocesar.py' completo.

POR QUE NO REPROCESAR TODO CADA DIA:
  Los anios 2023-2025 estan cerrados: nunca cambian. Reprocesarlos a diario
  es trabajo desperdiciado. Este script solo toca 2026 y deja el resto intacto.

USO:
  # 1) Descarga del portal OroraTech la info de 2026 a data/2026/
  #    (por-departamento renombrados, y/o el combinado diario)
  # 2) Corre:
  python actualizar_2026.py
  # 3) git add data/processed/ data/2026/ ; git commit ; git push

NOTA SOBRE DEDUPLICACION:
  Se deduplica por el campo 'id' del incendio (id estable de OroraTech).
  Si un incendio aparece en el archivo mensual Y en un combinado diario,
  se conserva una sola copia. Por defecto se conserva la PRIMERA aparicion;
  ver CONSERVAR_ULTIMO abajo si prefieres la version mas reciente.
============================================================================
"""

import time

import pandas as pd
import geopandas as gpd

# Reutilizamos TODA la maquinaria de preprocesar.py (no copiamos logica).
import preprocesar as pre

# ----------------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------------
ANIO_ACTUAL = "2026"

# Si un mismo 'id' aparece en varios archivos (ej. mensual + diario):
#   False -> conservar la primera aparicion (mas estable, recomendado)
#   True  -> conservar la ultima (util si el portal corrige datos)
CONSERVAR_ULTIMO = False
# ----------------------------------------------------------------------


def cargar_anio_actual():
    """Lee SOLO los geojson de data/<ANIO_ACTUAL>/ usando la logica compartida."""
    carpeta = pre.DIR_DATA / ANIO_ACTUAL
    if not carpeta.exists():
        print(f"[ERROR] No existe la carpeta {carpeta}")
        return None

    archivos = sorted(carpeta.glob("*.geojson"))
    if not archivos:
        print(f"[ERROR] No hay geojson en {carpeta}")
        return None

    print(f"  [{ANIO_ACTUAL}] {len(archivos)} archivos encontrados:")
    for a in archivos:
        print(f"      - {a.name}")

    registros = []
    sin_centroide = 0
    for archivo in archivos:
        regs, sc = pre.leer_archivo_incendios(archivo, ANIO_ACTUAL)
        registros.extend(regs)
        sin_centroide += sc

    if sin_centroide:
        print(f"  [aviso] {sin_centroide} incendios sin centroide valido (omitidos)")

    if not registros:
        print("[ERROR] No se extrajo ningun incendio con centroide valido.")
        return None

    return pre.registros_a_gdf(registros)


def cargar_historico_cerrado():
    """
    Lee la tabla ya procesada de anios cerrados y se queda con todo lo que
    NO sea el anio actual. Asi cada corrida reemplaza por completo 2026 con
    la version recien procesada (evita acumular duplicados entre corridas).
    """
    f_pq = pre.DIR_SALIDA / "incendios.parquet"
    f_csv = pre.DIR_SALIDA / "incendios.csv"

    if f_pq.exists():
        try:
            df = pd.read_parquet(f_pq)
            fuente = f_pq.name
        except Exception:
            df = pd.read_csv(f_csv)
            fuente = f_csv.name
    elif f_csv.exists():
        df = pd.read_csv(f_csv)
        fuente = f_csv.name
    else:
        print("  [aviso] No hay tabla previa. Corre preprocesar.py al menos una "
              "vez para tener 2023-2025. Continuo solo con 2026.")
        return pd.DataFrame()

    df["anio"] = df["anio"].astype(str)
    historico = df[df["anio"] != ANIO_ACTUAL].copy()
    print(f"  Historico leido de {fuente}: {len(df)} filas "
          f"({len(historico)} de anios != {ANIO_ACTUAL})")
    return historico


def main():
    t0 = time.time()
    print("=" * 60)
    print(f"ACTUALIZACION DIARIA {ANIO_ACTUAL} - INCENDIOS COLOMBIA")
    print("=" * 60)

    # Validaciones de limites (mismas rutas que preprocesar.py)
    for ruta in (pre.ARCHIVO_DEPTOS, pre.ARCHIVO_MUNIS):
        if not ruta.exists():
            print(f"[ERROR] No existe: {ruta}")
            return

    print(f"\n>> Cargando incendios de {ANIO_ACTUAL}...")
    gdf_2026 = cargar_anio_actual()
    if gdf_2026 is None:
        return
    print(f"   Incendios {ANIO_ACTUAL} con centroide: {len(gdf_2026)}")

    # Spatial join + regla de Bogota (logica compartida con preprocesar.py)
    gdf_2026 = pre.asignar_departamento_y_municipio(gdf_2026)

    # Reporte de no clasificados solo para 2026
    pre.reporte_no_clasificados(gdf_2026)

    # Pasar a DataFrame plano (sin geometria; el centroide ya esta en lon/lat)
    df_2026 = pd.DataFrame(gdf_2026.drop(columns="geometry"))
    df_2026["anio"] = df_2026["anio"].astype(str)

    # --- DEDUPLICACION dentro de 2026 (por si mensual + diario se solapan) ---
    antes = len(df_2026)
    if "id" in df_2026.columns:
        keep = "last" if CONSERVAR_ULTIMO else "first"
        # Conservar filas sin id (id nulo) tal cual; deduplicar solo las que tienen id
        con_id = df_2026[df_2026["id"].notna()].drop_duplicates(
            subset="id", keep=keep)
        sin_id = df_2026[df_2026["id"].isna()]
        df_2026 = pd.concat([con_id, sin_id], ignore_index=True)
        quitados = antes - len(df_2026)
        if quitados:
            print(f"\n>> Deduplicacion {ANIO_ACTUAL}: {quitados} duplicados por 'id' "
                  f"eliminados (conservando: {keep})")
    else:
        print("\n[aviso] No hay columna 'id'; no se pudo deduplicar.")

    # --- FUSION con anios cerrados ---
    print("\n>> Fusionando con anios cerrados...")
    historico = cargar_historico_cerrado()

    # Alinear columnas entre historico y 2026 antes de concatenar
    if len(historico):
        cols = list(dict.fromkeys(list(historico.columns) + list(df_2026.columns)))
        historico = historico.reindex(columns=cols)
        df_2026 = df_2026.reindex(columns=cols)
        combinado = pd.concat([historico, df_2026], ignore_index=True)
    else:
        combinado = df_2026

    print(f"   Total combinado (todos los anios): {len(combinado)} incendios")
    print("   Desglose por anio:")
    print(combinado["anio"].value_counts().sort_index().to_string()
          .replace("\n", "\n     "))

    # --- GUARDAR tabla principal combinada (mismos nombres que preprocesar.py) ---
    pre.DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    f_csv = pre.DIR_SALIDA / "incendios.csv"
    f_pq = pre.DIR_SALIDA / "incendios.parquet"

    combinado.to_csv(f_csv, index=False, encoding="utf-8")
    try:
        combinado.to_parquet(f_pq, index=False)
        extra_pq = f" y {f_pq.name}"
    except Exception:
        extra_pq = "  (parquet omitido: instala pyarrow para generarlo)"

    print(f"\n>> Tabla combinada guardada: {f_csv.name}{extra_pq}")
    print(f">> Carpeta de salida: {pre.DIR_SALIDA}")
    print("\n>> Recordatorio: los limites simplificados NO se regeneran aqui.")
    print("   Si los necesitas actualizar, corre: python preprocesar.py")
    print("\n>> Siguiente paso (manual):")
    print(f"   git add data/processed/ data/{ANIO_ACTUAL}/")
    print('   git commit -m "datos: actualizacion 2026"')
    print("   git push")
    print(f"\n>> Listo en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
