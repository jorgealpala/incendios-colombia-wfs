"""
============================================================================
DASHBOARD DE INCENDIOS - COLOMBIA  (OroraTech WildFire Solution)
============================================================================
Visor de 3 niveles (Nacional / Departamental / Municipal).
Metrica de confianza: confidence (valores numericos originales 0..1).
Filtro temporal: rango de fechas por 'oldest_acquisition' (inicio del incendio).

EJECUTAR LOCAL (Windows):
    python -m pip install -r requirements.txt
    python -m streamlit run app.py
============================================================================
"""

import json
import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

import reporte  # modulo local para generar el PDF

st.set_page_config(page_title="Incendios Colombia | OroraTech", page_icon=":fire:", layout="wide")

RAIZ = Path(__file__).resolve().parent
DIR_PROC = RAIZ / "data" / "processed"
F_PARQUET = DIR_PROC / "incendios.parquet"
F_CSV     = DIR_PROC / "incendios.csv"
F_DEPTOS = DIR_PROC / "departamentos_simplificado.geojson"
F_MUNIS  = DIR_PROC / "municipios_simplificado.geojson"

# --- confidence (unica metrica): valor numerico -> color en escala verde->rojo ---
ORDEN_CONF = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
COLOR_CONF = {
    "0.0": "#95a5a6",  # gris (datos insuficientes)
    "0.2": "#2ecc71",  # verde
    "0.4": "#f1c40f",  # amarillo
    "0.6": "#e67e22",  # naranja
    "0.8": "#e74c3c",  # rojo
    "1.0": "#c0392b",  # rojo oscuro
    "Sin dato": "#95a5a6",
}
ETIQUETAS = {"0.0": "0.0 (insuficiente)", "0.2": "0.2", "0.4": "0.4",
             "0.6": "0.6", "0.8": "0.8", "1.0": "1.0"}

# --- Colores FIJOS por año (consistentes en todas las gráficas) ---
# Cada año tiene su color asignado de forma estable: aunque se agreguen años
# nuevos (p. ej. 2019-2023 más adelante), los existentes NO cambian de color.
PALETA_ANIOS = ["#27ae60", "#2980b9", "#e74c3c", "#8e44ad", "#f39c12",
                "#16a085", "#c0392b", "#2c3e50", "#d35400", "#7f8c8d",
                "#e67e22", "#1abc9c"]
# Asignacion estable por año concreto (2019 en adelante)
ANIOS_REF = list(range(2019, 2031))
COLOR_ANIO = {a: PALETA_ANIOS[i % len(PALETA_ANIOS)] for i, a in enumerate(ANIOS_REF)}

def escala_color_anios(anios):
    """Devuelve (domain, range) de colores para una lista de años dada,
    usando el color FIJO de cada año (consistente entre todas las gráficas)."""
    dom = [str(a) for a in sorted(anios)]
    rng = [COLOR_ANIO.get(int(a), "#95a5a6") for a in sorted(anios)]
    return dom, rng

def cat_confidence(v):
    if v is None or pd.isna(v):
        return "Sin dato"
    return f"{float(v):.1f}"

@st.cache_data(show_spinner="Cargando incendios...")
def cargar_incendios():
    df = pd.read_parquet(F_PARQUET) if F_PARQUET.exists() else pd.read_csv(F_CSV)
    df["cat"] = df["confidence"].apply(cat_confidence) if "confidence" in df.columns else "Sin dato"
    # Fecha del incendio = oldest_acquisition (inicio); fallback a oldest_detection
    campo_fecha = "oldest_acquisition" if "oldest_acquisition" in df.columns else "oldest_detection"
    df["fecha"] = pd.to_datetime(df[campo_fecha], errors="coerce", utc=True)
    df["mes"] = df["fecha"].dt.month
    return df

@st.cache_data(show_spinner="Cargando mapa base...")
def cargar_geojson(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)

def feature_por_nombre(gj, campo, valor):
    """Devuelve un FeatureCollection con el unico feature cuyo campo == valor."""
    feats = [ft for ft in gj["features"] if ft["properties"].get(campo) == valor]
    return {"type": "FeatureCollection", "features": feats}

for k, v in {"nivel": "Nacional", "departamento": None, "municipio": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def ir_a_departamento(n):
    st.session_state.nivel = "Departamental"
    st.session_state.departamento = n
    st.session_state.municipio = None

if not F_DEPTOS.exists() or not (F_PARQUET.exists() or F_CSV.exists()):
    st.error("No se encontraron los datos procesados. Ejecuta primero "
             "`python preprocesar.py` para generar `data/processed/`.")
    st.stop()

df = cargar_incendios()
gj_deptos = cargar_geojson(F_DEPTOS)
gj_munis = cargar_geojson(F_MUNIS)

st.sidebar.title(":fire: Incendios Forestales")
st.sidebar.caption("Fuente de datos: WildFire Solution - Ororatech")

# ---- RANGO DE FECHAS (selector rapido de año + calendario fino) ----
fmin = df["fecha"].min()
fmax = df["fecha"].max()
fmin_d = fmin.date() if pd.notna(fmin) else dt.date(2016, 1, 1)
fmax_d = fmax.date() if pd.notna(fmax) else dt.date.today()

# Años disponibles en los datos (descendente: el mas reciente primero)
anios_disp = sorted(df["fecha"].dt.year.dropna().astype(int).unique(), reverse=True)
opciones_anio = ["Todos los años"] + [str(a) for a in anios_disp]
sel_anio = st.sidebar.selectbox(
    "Año", opciones_anio, index=0,
    help="Atajo: elige un año para ver el periodo completo. "
         "Para un rango más fino (días o meses), usa el calendario de abajo.")

# Segun el año elegido, definir el rango por defecto del calendario
if sel_anio == "Todos los años":
    # Rango por defecto institucional: 01/01/2024 a 31/05/2026,
    # siempre acotado a lo que realmente exista en los datos.
    ini_def = max(dt.date(2024, 1, 1), fmin_d)
    fin_def = min(dt.date(2026, 5, 31), fmax_d)
    if ini_def > fin_def:        # por si los datos no alcanzan ese rango
        ini_def, fin_def = fmin_d, fmax_d
    rango_def = (ini_def, fin_def)
else:
    a = int(sel_anio)
    ini_a = max(dt.date(a, 1, 1), fmin_d)
    fin_a = min(dt.date(a, 12, 31), fmax_d)
    rango_def = (ini_a, fin_a)

# La 'key' depende del año elegido y del rango de datos: al cambiar el año
# (o al cargar datos nuevos) el calendario se reinicia al periodo correcto.
key_fechas = f"rango_{sel_anio}_{fmin_d.isoformat()}_{fmax_d.isoformat()}"
rango = st.sidebar.date_input(
    "Rango de fechas (inicio del incendio)",
    value=rango_def, min_value=fmin_d, max_value=fmax_d,
    key=key_fechas,
    help="Filtra por la fecha de primera detección satelital (oldest_acquisition). "
         "Puedes ajustar el rango día a día aquí.")
# date_input puede devolver 1 fecha (mientras el usuario elige el rango) o 2.
if isinstance(rango, (tuple, list)) and len(rango) == 2:
    fecha_ini, fecha_fin = rango
elif isinstance(rango, (tuple, list)) and len(rango) == 1:
    fecha_ini, fecha_fin = rango[0], rango_def[1]
else:
    fecha_ini = fecha_fin = rango
etiqueta_periodo = (f"{fecha_ini:%d/%m/%Y} – {fecha_fin:%d/%m/%Y}"
                    if fecha_ini != fecha_fin else f"{fecha_ini:%d/%m/%Y}")
# Boton para recargar datos tras regenerar data/processed (limpia el cache)
if st.sidebar.button("🔄 Recargar datos", help="Úsalo si actualizaste los datos "
                     "(p. ej. agregaste un nuevo año) y no aparecen."):
    st.cache_data.clear()
    st.rerun()

# ============ NAVEGACION (incluye depto/municipio) ============
st.sidebar.divider()
st.sidebar.subheader("Navegación")
nivel = st.sidebar.radio("Nivel de vista", ["Nacional", "Departamental", "Municipal"],
                         index=["Nacional", "Departamental", "Municipal"].index(st.session_state.nivel))
st.session_state.nivel = nivel

# Filtrar por fecha (universo para navegacion)
mask_fecha = (df["fecha"].dt.date >= fecha_ini) & (df["fecha"].dt.date <= fecha_fin)
base = df[mask_fecha]
deptos_disponibles = sorted(base["departamento"].dropna().unique())

if nivel in ("Departamental", "Municipal") and deptos_disponibles:
    if st.session_state.departamento not in deptos_disponibles:
        st.session_state.departamento = deptos_disponibles[0]
    st.session_state.departamento = st.sidebar.selectbox(
        "Departamento", deptos_disponibles,
        index=deptos_disponibles.index(st.session_state.departamento))

if nivel == "Municipal" and st.session_state.departamento:
    munis_disp = sorted(base[base["departamento"] == st.session_state.departamento]
                        ["municipio"].dropna().unique())
    if munis_disp:
        if st.session_state.municipio not in munis_disp:
            st.session_state.municipio = munis_disp[0]
        st.session_state.municipio = st.sidebar.selectbox(
            "Municipio", munis_disp, index=munis_disp.index(st.session_state.municipio))

# ============ FILTRO DE CONFIANZA (solo confidence) ============
st.sidebar.divider()
st.sidebar.subheader("Confianza del incendio")
cats_presentes = [c for c in ORDEN_CONF if c in df["cat"].unique()]
mostrar_insuf = st.sidebar.checkbox("Mostrar incidentes con datos insuficientes", value=False)
cats_filtrables = [c for c in cats_presentes if c != "0.0"]
# Por defecto, activar solo los niveles de confianza alta (0.6, 0.8, 1.0).
# Si alguno no existe en los datos, simplemente no aparece.
default_conf = [c for c in ["0.6", "0.8", "1.0"] if c in cats_filtrables]
if not default_conf:                       # respaldo: si no hay altos, mostrar todos
    default_conf = cats_filtrables
cats_sel = st.sidebar.multiselect(
    "Filtrar niveles de confianza", cats_filtrables, default=default_conf,
    format_func=lambda x: ETIQUETAS.get(x, x))
if not cats_sel:
    cats_sel = cats_filtrables

# Aplicar filtro de categorias sobre la base ya filtrada por fecha
categorias_visibles = list(cats_sel)
if mostrar_insuf and "0.0" in cats_presentes:
    categorias_visibles.append("0.0")
dff = base[base["cat"].isin(categorias_visibles)]

# ---- REPORTE PDF (control; la generacion va al final cuando 'datos' existe) ----
st.sidebar.divider()
st.sidebar.subheader("Reporte PDF")
alcance = st.sidebar.radio(
    "Alcance del reporte",
    ["Vista actual", "Nacional completo"],
    help="'Vista actual' usa el nivel y filtros en pantalla. "
         "'Nacional completo' genera un resumen de todo el país.")
generar_clic = st.sidebar.button("Generar reporte PDF", width="stretch")

# ============ ENCABEZADO + KPIs ============
# Logo institucional + titulo (logo a la izquierda del titulo)
RUTA_LOGO = None
for cand in ["logo/ungrd-horizontal.png", "logo/UNGRD- Horizontal.png",
             "logo/UNGRD-Horizontal.png"]:
    if (RAIZ / cand).exists():
        RUTA_LOGO = str(RAIZ / cand)
        break

col_logo, col_tit = st.columns([1, 5])
with col_logo:
    if RUTA_LOGO:
        st.image(RUTA_LOGO, width="stretch")
with col_tit:
    st.markdown("## Incendios Forestales - Colombia")
    st.markdown(
        "<div style='font-size:14px;color:#555;line-height:1.3;'>"
        "Unidad Nacional para la Gestión del Riesgo de Desastres<br>"
        "Subdirección para el Conocimiento del Riesgo</div>",
        unsafe_allow_html=True)
st.write("")

if nivel == "Nacional":
    datos = dff
    st.subheader(f"Vista nacional · {etiqueta_periodo}")
elif nivel == "Departamental":
    datos = dff[dff["departamento"] == st.session_state.departamento]
    st.subheader(f"{st.session_state.departamento} · {etiqueta_periodo}")
else:
    datos = dff[(dff["departamento"] == st.session_state.departamento)
                & (dff["municipio"] == st.session_state.municipio)]
    st.subheader(f"{st.session_state.municipio} ({st.session_state.departamento}) · {etiqueta_periodo}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Eventos", f"{len(datos):,}")
c2.metric("Área total (ha)", f"{datos['area'].sum():,.0f}")
c3.metric("Área promedio (ha)", f"{datos['area'].mean():,.1f}" if len(datos) else "0")
n_alta = datos[datos["cat"].isin(["0.6", "0.8", "1.0"])].shape[0]
pct = (100 * n_alta / len(datos)) if len(datos) else 0
c4.metric("% confianza alta (≥0.6)", f"{pct:.0f}%")

# ============ MAPA (mas grande, ocupa todo el ancho) ============
def contorno_depto(nombre):
    fc = feature_por_nombre(gj_deptos, "DeNombre", nombre)
    return folium.GeoJson(fc, style_function=lambda x: {
        "color": "#1f3b57", "weight": 3, "fillOpacity": 0}, name="contorno")

def contorno_muni(nombre):
    fc = feature_por_nombre(gj_munis, "MpNombre", nombre)
    return folium.GeoJson(fc, style_function=lambda x: {
        "color": "#1f3b57", "weight": 3, "fillOpacity": 0}, name="contorno")

if nivel == "Nacional":
    conteo = dff.groupby("departamento").size()
    nombres_geo = [ft["properties"]["DeNombre"] for ft in gj_deptos["features"]]
    conteo = conteo.reindex(nombres_geo).fillna(0)
    import numpy as np
    vals = conteo.values
    cortes_np = np.unique(np.quantile(vals, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    cortes = [float(c) for c in cortes_np]
    usa_cuantiles = len(cortes) >= 3 and cortes[0] <= vals.min() and cortes[-1] >= vals.max()
    m = folium.Map(location=[4.6, -74.1], zoom_start=5, tiles="cartodbpositron")
    choro = folium.Choropleth(
        geo_data=gj_deptos, data=conteo,
        key_on="feature.properties.DeNombre",
        fill_color="YlOrRd", fill_opacity=0.7, line_opacity=0.3,
        nan_fill_color="#eeeeee",
        bins=cortes if usa_cuantiles else 6,
        legend_name="Frecuencia (total eventos)")
    choro.add_to(m)
    # Quitar la leyenda automatica de folium (se encima); usamos una propia debajo
    for key in list(choro._children.keys()):
        if key.startswith("color_map"):
            del choro._children[key]
    folium.GeoJson(gj_deptos,
                   style_function=lambda x: {"fillColor": "#00000000", "color": "#00000000",
                                             "weight": 0, "fillOpacity": 0},
                   highlight_function=lambda x: {"weight": 2, "color": "#333", "fillOpacity": 0.1},
                   tooltip=folium.GeoJsonTooltip(fields=["DeNombre"],
                                                 aliases=["Departamento:"], sticky=True)).add_to(m)
    salida = st_folium(m, width=None, height=620, returned_objects=["last_active_drawing"])
    # Leyenda propia de clases (legible, sin encimar etiquetas)
    paleta = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026", "#7a0019"]
    if usa_cuantiles and len(cortes) >= 3:
        items = []
        for i in range(len(cortes) - 1):
            ini, fin = cortes[i], cortes[i + 1]
            col = paleta[i] if i < len(paleta) else paleta[-1]
            items.append(
                f"<span style='display:inline-flex;align-items:center;margin-right:14px;'>"
                f"<span style='width:16px;height:16px;background:{col};"
                f"border:1px solid #ccc;border-radius:3px;margin-right:5px;'></span>"
                f"{ini:,.0f} – {fin:,.0f}</span>")
        st.markdown(
            "<div style='margin-top:-8px;'><b>Frecuencia (total eventos)</b> · clases por cuantiles<br>"
            + "".join(items) + "</div>", unsafe_allow_html=True)
    if salida and salida.get("last_active_drawing"):
        clic = salida["last_active_drawing"].get("properties", {}).get("DeNombre")
        if clic and clic in deptos_disponibles:
            ir_a_departamento(clic); st.rerun()

elif nivel == "Departamental":
    sub = datos.dropna(subset=["lat", "lon"])
    centro = [sub["lat"].mean(), sub["lon"].mean()] if len(sub) else [4.6, -74.1]
    m = folium.Map(location=centro, zoom_start=8, tiles="cartodbpositron")
    # 1) Heatmap primero (capa de fondo)
    if len(sub):
        HeatMap(sub[["lat", "lon"]].values.tolist(), radius=12, blur=18, min_opacity=0.3).add_to(m)
    # 2) Municipios DESPUES, para que queden encima y capturen el cursor (tooltip)
    munis_depto = feature_por_nombre(gj_munis, "Depto", st.session_state.departamento)
    if munis_depto["features"]:
        folium.GeoJson(
            munis_depto,
            style_function=lambda x: {"color": "#9aa6b2", "weight": 1,
                                      "fillColor": "#ffffff", "fillOpacity": 0.01,
                                      "dashArray": "3"},
            highlight_function=lambda x: {"color": "#1f3b57", "weight": 2,
                                          "fillOpacity": 0.08},
            tooltip=folium.GeoJsonTooltip(fields=["MpNombre"],
                                          aliases=["Municipio:"], sticky=True),
        ).add_to(m)
        # Etiquetas solo en municipios CON incendios en el periodo, priorizando
        # los de mayor numero de eventos (evita saturar deptos con muchos municipios).
        conteo_muni = datos.groupby("municipio").size().to_dict()
        # Maximo de etiquetas a mostrar; el resto queda solo con su contorno + tooltip
        MAX_ETIQUETAS = 15
        munis_top = sorted(conteo_muni, key=conteo_muni.get, reverse=True)[:MAX_ETIQUETAS]
        for ft in munis_depto["features"]:
            nombre_m = ft["properties"]["MpNombre"]
            if nombre_m not in munis_top:
                continue
            try:
                geom = ft["geometry"]
                coords = geom["coordinates"]
                ring = coords[0][0] if geom["type"] == "MultiPolygon" else coords[0]
                lons = [c[0] for c in ring]; lats = [c[1] for c in ring]
                clat, clon = sum(lats) / len(lats), sum(lons) / len(lons)
                folium.Marker(
                    [clat, clon],
                    icon=folium.DivIcon(
                        icon_size=(0, 0), icon_anchor=(0, 0),
                        html=(f"<div style='font-size:10px;font-weight:600;"
                              f"color:#3a4450;opacity:0.6;white-space:nowrap;"
                              f"text-shadow:0 0 3px #fff,0 0 3px #fff;"
                              f"transform:translate(-50%,-50%);'>"
                              f"{nombre_m}</div>")),
                ).add_to(m)
            except Exception:
                pass
    # 3) Contorno del departamento por encima
    contorno_depto(st.session_state.departamento).add_to(m)
    st.caption("Densidad de incendios. Se etiquetan los municipios con más eventos; "
               "pasa el cursor sobre cualquiera para ver su nombre.")
    salida = st_folium(m, width=None, height=620)

else:  # Municipal
    sub = datos.dropna(subset=["lat", "lon"])
    centro = [sub["lat"].mean(), sub["lon"].mean()] if len(sub) else [4.6, -74.1]
    m = folium.Map(location=centro, zoom_start=11, tiles="cartodbpositron")
    contorno_muni(st.session_state.municipio).add_to(m)
    for _, r in sub.iterrows():
        color = COLOR_CONF.get(r["cat"], "#95a5a6")
        radio = 4 + (r["area"] ** 0.5) / 4 if pd.notna(r["area"]) else 4
        folium.CircleMarker(location=[r["lat"], r["lon"]], radius=radio,
                            color=color, fill=True, fill_color=color, fill_opacity=0.7, weight=1,
                            tooltip=(f"ID {r['id']} · conf {r['cat']} · {r['area']:.1f} ha"
                                     if pd.notna(r['area']) else f"ID {r['id']} · conf {r['cat']}")).add_to(m)
    st.caption("El contorno marca el límite del municipio.")
    salida = st_folium(m, width=None, height=620)

# ============ GRAFICAS: dos por fila, debajo del mapa ============
st.divider()
st.markdown("### Análisis")

def barras_top(data, campo, titulo, color, valor="eventos"):
    st.markdown(f"**{titulo}**")
    if valor == "eventos":
        g = data.groupby(campo).size().sort_values(ascending=False).head(10).reset_index()
        g.columns = [campo, "valor"]; ytt = "Eventos"
    else:
        g = data.groupby(campo)["area"].sum().sort_values(ascending=False).head(10).reset_index()
        g.columns = [campo, "valor"]; ytt = "Área (ha)"
    ch = (alt.Chart(g).mark_bar(color=color).encode(
        x=alt.X("valor:Q", title=ytt),
        y=alt.Y(f"{campo}:N", sort="-x", title=None),
        tooltip=[campo, alt.Tooltip("valor:Q", title=ytt, format=",.0f")]
    ).properties(height=280))
    st.altair_chart(ch, width="stretch")

# Fila 1: distribucion por confianza + eventos por mes
g1, g2 = st.columns(2)
with g1:
    st.markdown("**Distribución por nivel de confianza**")
    dist = datos.groupby("cat").size().reindex(cats_presentes).fillna(0).reset_index()
    dist.columns = ["nivel", "eventos"]
    ch = (alt.Chart(dist).mark_bar().encode(
        x=alt.X("nivel:N", sort=cats_presentes, title="Nivel de confianza"),
        y=alt.Y("eventos:Q", title="Eventos"),
        color=alt.Color("nivel:N", scale=alt.Scale(domain=list(COLOR_CONF.keys()),
                                                    range=list(COLOR_CONF.values())), legend=None),
        tooltip=["nivel", "eventos"]).properties(height=280))
    st.altair_chart(ch, width="stretch")
with g2:
    st.markdown("**Eventos por mes (comparativo por año)**")
    if datos["fecha"].notna().any():
        meses_abrev = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
                       7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}
        orden_meses = list(meses_abrev.values())
        tmp = datos.dropna(subset=["fecha"]).copy()
        tmp["anio"] = tmp["fecha"].dt.year.astype(int)
        tmp["mes"] = tmp["fecha"].dt.month.astype(int)
        # Conteo por año y mes
        serie = (tmp.groupby(["anio", "mes"]).size().reset_index(name="eventos"))
        serie["mes_nombre"] = serie["mes"].map(meses_abrev)
        serie["anio"] = serie["anio"].astype(str)
        # Linea de promedio entre años (promedio de cada mes a traves de los años)
        prom = (serie.groupby("mes")["eventos"].mean().reindex(range(1, 13))
                .reset_index())
        prom["mes_nombre"] = prom["mes"].map(meses_abrev)
        prom["serie"] = "Promedio"

        base_x = alt.X("mes_nombre:N", sort=orden_meses, title="Mes",
                       axis=alt.Axis(labelAngle=0))
        # Lineas por año, con color FIJO por año (consistente con las
        # graficas de analisis comparativo y estable al agregar años nuevos).
        anios_ord = sorted(serie["anio"].unique())
        dom_anios, rng_anios = escala_color_anios(anios_ord)
        lineas = (alt.Chart(serie).mark_line(point=True).encode(
            x=base_x,
            y=alt.Y("eventos:Q", title="Eventos"),
            color=alt.Color("anio:N", title="Año",
                            scale=alt.Scale(domain=dom_anios, range=rng_anios)),
            tooltip=[alt.Tooltip("anio:N", title="Año"),
                     alt.Tooltip("mes_nombre:N", title="Mes"),
                     alt.Tooltip("eventos:Q", title="Eventos")]))
        # Linea de promedio (negra, punteada)
        linea_prom = (alt.Chart(prom).mark_line(
            strokeDash=[6, 4], color="#000000", point=False).encode(
            x=base_x,
            y=alt.Y("eventos:Q"),
            tooltip=[alt.Tooltip("mes_nombre:N", title="Mes"),
                     alt.Tooltip("eventos:Q", title="Promedio", format=",.0f")]))
        ch = (lineas + linea_prom).properties(height=280).resolve_scale(color="independent")
        st.altair_chart(ch, width="stretch")
        st.caption("Cada línea de color es un año; la línea negra punteada es el promedio mensual entre años.")
    else:
        st.info("Sin datos de fecha para el periodo seleccionado.")

# Fila: serie DIARIA a todo el ancho (rango seleccionado por el usuario)
st.markdown("**Eventos por día (rango seleccionado)**")
if datos["fecha"].notna().any():
    diaria = (datos.dropna(subset=["fecha"])
              .assign(dia=lambda d: d["fecha"].dt.floor("D"))
              .groupby("dia").size().reset_index(name="eventos"))
    if len(diaria):
        # Rellenar dias sin eventos con 0 para una serie continua
        rango_dias = pd.date_range(diaria["dia"].min(), diaria["dia"].max(),
                                   freq="D", tz="UTC")
        diaria = (diaria.set_index("dia").reindex(rango_dias, fill_value=0)
                  .rename_axis("dia").reset_index())
        # Promedio movil de 7 dias
        diaria["media7"] = diaria["eventos"].rolling(7, min_periods=1, center=True).mean()

        base = alt.Chart(diaria).encode(
            x=alt.X("dia:T", title="Fecha"))
        linea_dia = base.mark_line(color="#bdc3c7", opacity=0.8).encode(
            y=alt.Y("eventos:Q", title="Eventos"),
            tooltip=[alt.Tooltip("dia:T", title="Día"),
                     alt.Tooltip("eventos:Q", title="Eventos")])
        linea_media = base.mark_line(color="#e74c3c", strokeWidth=2).encode(
            y=alt.Y("media7:Q"),
            tooltip=[alt.Tooltip("dia:T", title="Día"),
                     alt.Tooltip("media7:Q", title="Media 7 días", format=",.1f")])
        ch_dia = (linea_dia + linea_media).properties(height=300)
        st.altair_chart(ch_dia, width="stretch")
        st.caption("Línea gris: eventos diarios. Línea roja: promedio móvil de 7 días (suaviza el ruido).")
    else:
        st.info("Sin datos diarios para el periodo seleccionado.")
else:
    st.info("Sin datos de fecha para el periodo seleccionado.")

# ============ Análisis por AÑO (tortas + tendencia) ============
# Usa 'datos' (filtrado por fecha, confianza Y nivel geografico seleccionado),
# para que al navegar a un departamento o municipio estas graficas reflejen
# esa zona, igual que los KPIs y las demas graficas.
serie_anual = (datos.dropna(subset=["fecha"])
               .assign(anio=lambda d: d["fecha"].dt.year.astype(int)))
if len(serie_anual):
    resumen = (serie_anual.groupby("anio")
               .agg(eventos=("id", "size"), area=("area", "sum"))
               .reset_index().sort_values("anio"))
    resumen["anio_str"] = resumen["anio"].astype(str)
    # Variacion % de eventos respecto al año anterior
    resumen["var_pct"] = resumen["eventos"].pct_change() * 100

    st.markdown("**Análisis comparativo por año**")
    pa, pb, pc = st.columns(3)

    # Escala de color FIJA por año, igual que en la gráfica de eventos por mes.
    dom_a, rng_a = escala_color_anios(resumen["anio"].tolist())
    escala_anio = alt.Scale(domain=dom_a, range=rng_a)

    # (a) Tortas: eventos por año y área por año
    with pa:
        torta_ev = (alt.Chart(resumen).mark_arc(innerRadius=40).encode(
            theta=alt.Theta("eventos:Q"),
            color=alt.Color("anio_str:N", title="Año", scale=escala_anio),
            tooltip=[alt.Tooltip("anio_str:N", title="Año"),
                     alt.Tooltip("eventos:Q", title="Eventos", format=",")]
        ).properties(height=220, title="Eventos por año"))
        st.altair_chart(torta_ev, width="stretch")

        torta_ar = (alt.Chart(resumen).mark_arc(innerRadius=40).encode(
            theta=alt.Theta("area:Q"),
            color=alt.Color("anio_str:N", title="Año", scale=escala_anio),
            tooltip=[alt.Tooltip("anio_str:N", title="Año"),
                     alt.Tooltip("area:Q", title="Área (ha)", format=",.0f")]
        ).properties(height=220, title="Área quemada por año"))
        st.altair_chart(torta_ar, width="stretch")

    # (b) Barras de eventos por año con variacion % (tendencia)
    with pb:
        st.caption("Eventos por año y su variación frente al año anterior")
        barras_ev = (alt.Chart(resumen).mark_bar().encode(
            x=alt.X("anio_str:N", title="Año"),
            y=alt.Y("eventos:Q", title="Eventos"),
            color=alt.Color("anio_str:N", scale=escala_anio, legend=None),
            tooltip=[alt.Tooltip("anio_str:N", title="Año"),
                     alt.Tooltip("eventos:Q", title="Eventos", format=","),
                     alt.Tooltip("var_pct:Q", title="Var. % vs año anterior", format="+.1f")]
        ).properties(height=300))
        # Etiqueta con la variacion % encima de cada barra (desde el 2do año)
        etiquetas = (alt.Chart(resumen[resumen["var_pct"].notna()])
                     .mark_text(dy=-8, fontSize=11, fontWeight="bold")
                     .encode(x=alt.X("anio_str:N"), y=alt.Y("eventos:Q"),
                             text=alt.Text("var_pct:Q", format="+.0f"),
                             color=alt.condition(alt.datum.var_pct >= 0,
                                                 alt.value("#c0392b"), alt.value("#27ae60"))))
        st.altair_chart(barras_ev + etiquetas, width="stretch")

    # (c) Barras de area quemada por año
    with pc:
        st.caption("Superficie quemada por año (hectáreas)")
        barras_ar = (alt.Chart(resumen).mark_bar().encode(
            x=alt.X("anio_str:N", title="Año"),
            y=alt.Y("area:Q", title="Área (ha)"),
            color=alt.Color("anio_str:N", scale=escala_anio, legend=None),
            tooltip=[alt.Tooltip("anio_str:N", title="Año"),
                     alt.Tooltip("area:Q", title="Área (ha)", format=",.0f")]
        ).properties(height=300))
        st.altair_chart(barras_ar, width="stretch")

# ============ TOPS (al final, antes de la tabla) ============
g3, g4 = st.columns(2)
if nivel == "Nacional":
    with g3: barras_top(dff, "departamento", "Top 10 departamentos por eventos", "#e74c3c", "eventos")
    with g4: barras_top(dff, "departamento", "Top 10 departamentos por área (ha)", "#c0392b", "area")
elif nivel == "Departamental":
    with g3: barras_top(datos, "municipio", "Top 10 municipios por eventos", "#e74c3c", "eventos")
    with g4: barras_top(datos, "municipio", "Top 10 municipios por área (ha)", "#c0392b", "area")
else:
    with g3:
        st.markdown("**Distribución de área por evento (ha)**")
        if len(datos):
            ch = (alt.Chart(datos.assign(a=datos["area"].clip(upper=datos["area"].quantile(0.98))))
                  .mark_bar(color="#e67e22").encode(
                      x=alt.X("a:Q", bin=alt.Bin(maxbins=25), title="Área (ha)"),
                      y=alt.Y("count():Q", title="Eventos")).properties(height=280))
            st.altair_chart(ch, width="stretch")
    with g4:
        st.markdown("**Eventos por satélite**")
        if "satellites" in datos.columns and len(datos):
            sat = (datos.assign(sat=datos["satellites"].fillna("").str.split(","))
                   .explode("sat"))
            sat = sat[sat["sat"] != ""]
            top = sat.groupby("sat").size().sort_values(ascending=False).head(10).reset_index()
            top.columns = ["satelite", "eventos"]
            ch = (alt.Chart(top).mark_bar(color="#16a085").encode(
                x=alt.X("eventos:Q", title="Eventos"),
                y=alt.Y("satelite:N", sort="-x", title=None),
                tooltip=["satelite", "eventos"]).properties(height=280))
            st.altair_chart(ch, width="stretch")

st.divider()
st.markdown("#### Tabla de eventos")
cols = ["id", "departamento", "municipio", "area", "confidence", "fire_confidence",
        "num_fires", "lifetime", "satellites", "oldest_acquisition", "newest_detection"]
cols = [c for c in cols if c in datos.columns]
st.dataframe(datos[cols].sort_values("area", ascending=False), width="stretch", hide_index=True)
st.download_button("Descargar esta selección (CSV)",
                   datos[cols].to_csv(index=False).encode("utf-8"),
                   file_name="incendios_seleccion.csv", mime="text/csv")

# ============ GENERACION DEL REPORTE PDF (al final: 'datos' ya existe) ============
@st.cache_data(show_spinner=False)
def cargar_gdf(ruta):
    import geopandas as gpd
    return gpd.read_file(ruta)

if generar_clic:
    with st.spinner("Generando reporte PDF..."):
        try:
            gdf_dep = cargar_gdf(F_DEPTOS)
            if alcance == "Nacional completo":
                pdf_bytes = reporte.generar_pdf(
                    dff, "Nacional", "Vista nacional", etiqueta_periodo,
                    cats_presentes, gdf_deptos=gdf_dep,
                    conteo_por_depto=dff.groupby("departamento").size().to_dict(),
                    ruta_logo=RUTA_LOGO)
                fname = "reporte_nacional.pdf"
            elif nivel == "Nacional":
                pdf_bytes = reporte.generar_pdf(
                    datos, "Nacional", "Vista nacional", etiqueta_periodo,
                    cats_presentes, gdf_deptos=gdf_dep,
                    conteo_por_depto=datos.groupby("departamento").size().to_dict(),
                    ruta_logo=RUTA_LOGO)
                fname = "reporte_nacional.pdf"
            elif nivel == "Departamental":
                limite = gdf_dep[gdf_dep["DeNombre"] == st.session_state.departamento]
                pdf_bytes = reporte.generar_pdf(
                    datos, "Departamental", st.session_state.departamento,
                    etiqueta_periodo, cats_presentes, gdf_limite=limite,
                    ruta_logo=RUTA_LOGO)
                fname = f"reporte_{st.session_state.departamento}.pdf"
            else:
                gdf_mun = cargar_gdf(F_MUNIS)
                limite = gdf_mun[gdf_mun["MpNombre"] == st.session_state.municipio]
                pdf_bytes = reporte.generar_pdf(
                    datos, "Municipal",
                    f"{st.session_state.municipio} ({st.session_state.departamento})",
                    etiqueta_periodo, cats_presentes, gdf_limite=limite,
                    ruta_logo=RUTA_LOGO)
                fname = f"reporte_{st.session_state.municipio}.pdf"
            st.session_state["pdf_bytes"] = pdf_bytes
            st.session_state["pdf_fname"] = fname
        except Exception as e:
            st.error(f"No se pudo generar el reporte: {e}")

if st.session_state.get("pdf_bytes"):
    st.success("Reporte listo para descargar.")
    st.download_button(
        "⬇️ Descargar reporte PDF", st.session_state["pdf_bytes"],
        file_name=st.session_state.get("pdf_fname", "reporte.pdf"),
        mime="application/pdf")

# ============ PIE DE PAGINA: descargo, autoria y creditos ============
st.divider()
st.markdown("#### Descargo de responsabilidad")
st.markdown(
    "<div style='font-size:13px;color:#444;text-align:justify;'>"
    "La UNGRD, a través de la SCR, comparte la siguiente información, "
    "deslindándose de cualquier responsabilidad sobre el uso que se le dé. "
    "Los datos presentados han sido generados a partir de información extraída "
    "de la plataforma WFS de Ororatech. Este visor se suministra con el propósito "
    "de orientar la toma de decisiones; sin embargo, su correcto uso y aplicación "
    "son responsabilidad exclusiva de cada persona o entidad territorial."
    "</div>", unsafe_allow_html=True)

st.write("")
cred1, cred2 = st.columns(2)
with cred1:
    st.markdown(
        "<div style='font-size:13px;color:#444;'>"
        "<b>Realizado por:</b> Jorge Alpala<br>"
        "(jorge.alpala@gestiondelriesgo.gov.co, UNGRD - SCR)<br>"
        "<b>Elaboración:</b> Mayo de 2026</div>",
        unsafe_allow_html=True)
with cred2:
    st.markdown(
        "<div style='font-size:13px;color:#444;'>"
        "<b>Fuentes y créditos:</b><br>"
        "Datos: OroraTech WildFire Solution (WFS) · "
        "Límites: DIVIPOLA - DANE.<br>"
        "Desarrollado con Python, Streamlit, Folium/Leaflet, GeoPandas, "
        "Shapely, Pandas, Altair, Matplotlib y ReportLab. "
        "Mapas base: OpenStreetMap · CARTO."
        "</div>", unsafe_allow_html=True)
