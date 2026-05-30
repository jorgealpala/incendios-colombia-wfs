"""
============================================================================
PREPROCESAMIENTO - Dashboard de incendios Colombia (OroraTech WFS)
============================================================================
Este script se corre UNA SOLA VEZ en tu maquina (o cada vez que agregues
datos nuevos). Genera los archivos livianos que leera el dashboard de
Streamlit, para que cargue rapido y entre en GitHub.

QUE HACE:
  1. Recorre todos los GeoJSON de incendios en data/<anio>/
  2. Extrae el 'centroid' y los atributos de cada incendio
  3. Asigna cada incendio a su DEPARTAMENTO y MUNICIPIO por ubicacion
     (spatial join contra los limites DIVIPOLA)
  4. Maneja BOGOTA como unidad propia (igual que la plataforma OroraTech):
     los incendios del archivo 'bogota.geojson' se etiquetan como Bogota D.C.
  5. Reporta cuantos incendios quedaron SIN clasificar (frontera/rios)
  6. Simplifica los limites para que pesen poco (GitHub < 100 MB/archivo)
  7. Guarda todo en data/processed/

ESTRUCTURA ESPERADA DEL PROYECTO:
  incendios-colombia-wfs/
  ├── data/
  │   ├── 2024/  (33 geojson de incendios)
  │   ├── 2025/  (33 geojson de incendios)
  │   └── processed/   <- lo genera este script
  ├── divipola/
  │   ├── departamentos.geojson
  │   └── municipios.geojson
  └── preprocesar.py   <- este archivo

USO:
  pip install geopandas shapely pandas
  python preprocesar.py
============================================================================
"""

import json
import time
import unicodedata
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ----------------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------------
# Carpeta raiz del proyecto (donde esta este script).
RAIZ = Path(__file__).resolve().parent

DIR_DATA      = RAIZ / "data"
DIR_DIVIPOLA  = RAIZ / "divipola"
DIR_SALIDA    = DIR_DATA / "processed"

ARCHIVO_DEPTOS = DIR_DIVIPOLA / "departamentos.geojson"
ARCHIVO_MUNIS  = DIR_DIVIPOLA / "municipios.geojson"

# Anios a procesar: subcarpetas dentro de data/ que son numeros (2024, 2025...)
# Se detectan automaticamente, pero puedes fijarlos manualmente si quieres:
# ANIOS = ["2024", "2025"]
ANIOS = None  # None = autodetectar

# Tolerancia de simplificacion de geometria (en grados).
# 0.001 ~ 100 m. Mas alto = archivo mas liviano pero menos detalle.
TOLERANCIA_SIMPLIFICACION = 0.001

# Como se llama el archivo de Bogota (sin extension) en tus datos.
NOMBRE_ARCHIVO_BOGOTA = "bogota"
CODIGO_MUNI_BOGOTA = "11001"   # codigo DANE de Bogota en municipios.geojson
ETIQUETA_BOGOTA = "Bogotá, D.C."
# ----------------------------------------------------------------------


def quitar_tildes(texto: str) -> str:
    """Normaliza para comparar nombres sin importar tildes/mayusculas."""
    if not isinstance(texto, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", texto)
    base = "".join(c for c in nfkd if not unicodedata.combining(c))
    return base.replace("ñ", "n").replace("Ñ", "N").lower().strip()


def detectar_anios():
    if ANIOS:
        return ANIOS
    anios = []
    for sub in sorted(DIR_DATA.iterdir()):
        if sub.is_dir() and sub.name.isdigit():
            anios.append(sub.name)
    return anios


def cargar_incendios(anios):
    """
    Lee todos los GeoJSON de incendios y devuelve un GeoDataFrame de puntos
    (centroides) con sus atributos, el anio y el archivo de origen.
    """
    registros = []
    sin_centroide = 0

    for anio in anios:
        carpeta = DIR_DATA / anio
        archivos = sorted(carpeta.glob("*.geojson"))
        print(f"  [{anio}] {len(archivos)} archivos de incendios")

        for archivo in archivos:
            origen = archivo.stem  # ej: 'choco', 'bogota'
            with open(archivo, encoding="utf-8") as f:
                gj = json.load(f)

            for ft in gj.get("features", []):
                p = ft.get("properties", {})
                c = p.get("centroid")
                # El centroid debe ser un dict con lat/lon validos
                if not (isinstance(c, dict) and c.get("lat") is not None
                        and c.get("lon") is not None):
                    sin_centroide += 1
                    continue

                registros.append({
                    "id": p.get("id"),
                    "anio": anio,
                    "archivo_origen": origen,
                    "lon": c["lon"],
                    "lat": c["lat"],
                    "area": p.get("area"),
                    "confidence": p.get("confidence"),
                    "fire_confidence": p.get("fire_confidence"),
                    "num_fires": p.get("num_fires"),
                    "lifetime": p.get("lifetime"),
                    "type_string": p.get("type_string"),
                    "cause_string": p.get("cause_string"),
                    "sub_area_name": p.get("sub_area_name"),
                    "oldest_detection": p.get("oldest_detection"),
                    "newest_detection": p.get("newest_detection"),
                    "oldest_acquisition": p.get("oldest_acquisition"),
                    "newest_acquisition": p.get("newest_acquisition"),
                    # algorithms y satellites son listas -> a texto
                    "algorithms": ",".join(p.get("algorithms", []) or []),
                    "satellites": ",".join(p.get("satellites", []) or []),
                })

    if sin_centroide:
        print(f"  [aviso] {sin_centroide} incendios sin centroide valido (omitidos)")

    df = pd.DataFrame(registros)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )
    return gdf


def asignar_departamento_y_municipio(gdf_fires):
    """
    Spatial join de los incendios contra limites departamentales y municipales.
    Maneja Bogota como unidad propia segun el archivo de origen.
    """
    print("\n>> Spatial join contra limites...")

    deptos = gpd.read_file(ARCHIVO_DEPTOS)[["DeNombre", "DeCodigo", "geometry"]]
    munis = gpd.read_file(ARCHIVO_MUNIS)[["MpNombre", "MpCodigo", "Depto", "geometry"]]

    # Asegurar mismo sistema de coordenadas
    deptos = deptos.to_crs("EPSG:4326")
    munis = munis.to_crs("EPSG:4326")

    # --- Join departamental ---
    j = gpd.sjoin(gdf_fires, deptos, how="left", predicate="within")
    # Si un punto cae en 2 poligonos (raro), quedarse con el primero
    j = j[~j.index.duplicated(keep="first")].copy()
    j = j.rename(columns={"DeNombre": "departamento", "DeCodigo": "cod_depto"})
    j = j.drop(columns=[c for c in j.columns if c.startswith("index_right")],
               errors="ignore")

    # --- Join municipal ---
    j = gpd.sjoin(j, munis, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].copy()
    j = j.rename(columns={"MpNombre": "municipio", "MpCodigo": "cod_muni"})
    j = j.drop(columns=[c for c in j.columns if c.startswith("index_right")],
               errors="ignore")

    # --- BOGOTA: respetar la separacion de la plataforma ---
    # Todo incendio que venga del archivo 'bogota' se etiqueta como Bogota,
    # sin importar que geograficamente caiga dentro de Cundinamarca.
    es_bogota = j["archivo_origen"] == NOMBRE_ARCHIVO_BOGOTA
    j.loc[es_bogota, "departamento"] = ETIQUETA_BOGOTA
    j.loc[es_bogota, "cod_depto"] = CODIGO_MUNI_BOGOTA
    j.loc[es_bogota, "municipio"] = ETIQUETA_BOGOTA
    j.loc[es_bogota, "cod_muni"] = CODIGO_MUNI_BOGOTA
    print(f"   Bogota: {es_bogota.sum()} incendios etiquetados como unidad propia")

    return j


def reporte_no_clasificados(gdf):
    """Imprime y devuelve los incendios que no quedaron asignados a depto."""
    sin_depto = gdf[gdf["departamento"].isna()]
    total = len(gdf)
    n = len(sin_depto)
    print("\n" + "=" * 60)
    print("REPORTE DE CLASIFICACION")
    print("=" * 60)
    print(f"  Total incendios procesados : {total}")
    print(f"  Asignados a un departamento: {total - n}  ({100*(total-n)/total:.1f}%)")
    print(f"  SIN clasificar             : {n}  ({100*n/total:.1f}%)")
    if n:
        print("\n  Los no clasificados suelen estar sobre fronteras, rios o costa.")
        print("  Desglose por archivo de origen:")
        print(sin_depto["archivo_origen"].value_counts().to_string()
              .replace("\n", "\n     "))
    print("=" * 60)
    return sin_depto


def simplificar_limites():
    """Genera versiones livianas de los limites para el dashboard + Bogota."""
    print("\n>> Simplificando limites para el dashboard...")
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    deptos = gpd.read_file(ARCHIVO_DEPTOS).to_crs("EPSG:4326")
    munis = gpd.read_file(ARCHIVO_MUNIS).to_crs("EPSG:4326")

    # Extraer poligono de Bogota desde municipios y agregarlo a departamentos
    bogota = munis[munis["MpCodigo"] == CODIGO_MUNI_BOGOTA].copy()
    if len(bogota):
        bog_row = gpd.GeoDataFrame({
            "DeNombre": [ETIQUETA_BOGOTA],
            "DeCodigo": [CODIGO_MUNI_BOGOTA],
            "geometry": [bogota.geometry.iloc[0]],
        }, crs="EPSG:4326")
        deptos_out = pd.concat(
            [deptos[["DeNombre", "DeCodigo", "geometry"]], bog_row],
            ignore_index=True)
        deptos_out = gpd.GeoDataFrame(deptos_out, crs="EPSG:4326")
        print(f"   Bogota agregada como unidad en el mapa nacional")
    else:
        deptos_out = deptos[["DeNombre", "DeCodigo", "geometry"]].copy()

    # Simplificar geometrias
    deptos_out["geometry"] = deptos_out.geometry.simplify(
        TOLERANCIA_SIMPLIFICACION, preserve_topology=True)
    munis_out = munis[["MpNombre", "MpCodigo", "Depto", "geometry"]].copy()
    munis_out["geometry"] = munis_out.geometry.simplify(
        TOLERANCIA_SIMPLIFICACION, preserve_topology=True)

    f_dep = DIR_SALIDA / "departamentos_simplificado.geojson"
    f_mun = DIR_SALIDA / "municipios_simplificado.geojson"
    deptos_out.to_file(f_dep, driver="GeoJSON")
    munis_out.to_file(f_mun, driver="GeoJSON")

    mb = lambda p: round(p.stat().st_size / 1e6, 1)
    print(f"   {f_dep.name}: {mb(f_dep)} MB")
    print(f"   {f_mun.name}: {mb(f_mun)} MB")


def main():
    t0 = time.time()
    print("=" * 60)
    print("PREPROCESAMIENTO DE INCENDIOS - COLOMBIA")
    print("=" * 60)

    # Validaciones basicas
    for ruta in (ARCHIVO_DEPTOS, ARCHIVO_MUNIS):
        if not ruta.exists():
            print(f"[ERROR] No existe: {ruta}")
            return

    anios = detectar_anios()
    if not anios:
        print(f"[ERROR] No se encontraron carpetas de anio en {DIR_DATA}")
        return
    print(f"Anios detectados: {', '.join(anios)}\n>> Cargando incendios...")

    gdf_fires = cargar_incendios(anios)
    print(f"   Total incendios con centroide: {len(gdf_fires)}")

    gdf = asignar_departamento_y_municipio(gdf_fires)

    sin_clasificar = reporte_no_clasificados(gdf)

    # Guardar tabla principal (sin geometria; el centroide ya esta en lon/lat)
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(gdf.drop(columns="geometry"))

    f_csv = DIR_SALIDA / "incendios.csv"
    f_pq = DIR_SALIDA / "incendios.parquet"
    df_out.to_csv(f_csv, index=False, encoding="utf-8")
    try:
        df_out.to_parquet(f_pq, index=False)
        extra_pq = f" y {f_pq.name}"
    except Exception:
        extra_pq = "  (parquet omitido: instala pyarrow para generarlo)"

    # Guardar reporte de no clasificados aparte (por si quieres revisarlos)
    if len(sin_clasificar):
        f_nc = DIR_SALIDA / "incendios_sin_clasificar.csv"
        pd.DataFrame(sin_clasificar.drop(columns="geometry")).to_csv(
            f_nc, index=False, encoding="utf-8")
        print(f"\n   No clasificados guardados en: {f_nc.name}")

    simplificar_limites()

    print(f"\n>> Tabla principal: {f_csv.name}{extra_pq}")
    print(f">> Todo guardado en: {DIR_SALIDA}")
    print(f">> Listo en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
