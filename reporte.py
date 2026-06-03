"""
============================================================================
GENERADOR DE REPORTE PDF - Dashboard de incendios Colombia
============================================================================
Crea un reporte ejecutivo en PDF con:
  - Portada con KPIs
  - Mapa estatico (matplotlib + GeoJSON, sin depender de internet)
  - Graficas (distribucion por confianza, eventos por mes, tops)
  - Tabla de los principales eventos

El alcance lo decide quien llama la funcion (nivel + filtros ya aplicados).
============================================================================
"""

import io
import datetime as dt

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla, necesario en servidor/nube
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

# Colores de la escala de confianza (coinciden con el dashboard)
COLOR_CONF = {
    "0.0": "#95a5a6", "0.2": "#2ecc71", "0.4": "#f1c40f",
    "0.6": "#e67e22", "0.8": "#e74c3c", "1.0": "#c0392b",
}


# ----------------------------------------------------------------------
# GENERADORES DE IMAGENES (matplotlib -> buffer PNG)
# ----------------------------------------------------------------------
def _fig_a_buffer(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def mapa_nacional(gdf_deptos, conteo_por_depto):
    """Coropletico de departamentos por numero de eventos (clases por cuantiles)."""
    import numpy as np
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    g = gdf_deptos.copy()
    g["eventos"] = g["DeNombre"].map(conteo_por_depto).fillna(0)
    cmap = LinearSegmentedColormap.from_list("fuego", ["#ffffcc", "#fd8d3c", "#bd0026"])
    vals = g["eventos"].values
    # Cortes por cuantiles (5 clases); si no hay suficiente variedad, escala lineal
    try:
        cortes = np.unique(np.quantile(vals, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        if len(cortes) >= 3:
            norm = mcolors.BoundaryNorm(cortes, cmap.N)
            g.plot(column="eventos", cmap=cmap, norm=norm, linewidth=0.4,
                   edgecolor="#888", ax=ax, legend=True,
                   legend_kwds={"label": "Frecuencia (eventos)", "shrink": 0.5,
                                "ticks": cortes})
        else:
            raise ValueError
    except Exception:
        g.plot(column="eventos", cmap=cmap, linewidth=0.4, edgecolor="#888",
               ax=ax, legend=True, legend_kwds={"label": "Eventos", "shrink": 0.5})
    ax.set_title("Frecuencia de eventos por departamento (cuantiles)",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
    return _fig_a_buffer(fig)


def mapa_puntos(gdf_limite, df_puntos, titulo):
    """Contorno de la zona + puntos de incendio coloreados por confianza."""
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    if gdf_limite is not None and len(gdf_limite):
        gdf_limite.boundary.plot(ax=ax, color="#1f3b57", linewidth=1.2)
    if len(df_puntos):
        for cat, color in COLOR_CONF.items():
            sub = df_puntos[df_puntos["cat"] == cat]
            if len(sub):
                ax.scatter(sub["lon"], sub["lat"], s=18, c=color,
                           alpha=0.6, edgecolors="none", label=cat)
        ax.legend(title="Confianza", fontsize=7, title_fontsize=8,
                  loc="upper right", framealpha=0.9)
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.set_aspect("equal", adjustable="datalim")
    return _fig_a_buffer(fig)


def grafica_distribucion(datos, cats_presentes):
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    conteo = datos.groupby("cat").size().reindex(cats_presentes).fillna(0)
    barras = ax.bar(range(len(cats_presentes)), conteo.values,
                    color=[COLOR_CONF.get(c, "#999") for c in cats_presentes])
    ax.set_xticks(range(len(cats_presentes)))
    ax.set_xticklabels(cats_presentes)
    ax.set_ylabel("Eventos")
    ax.set_title("Distribución por nivel de confianza", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_a_buffer(fig)


MESES_ABREV = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}

# Colores FIJOS por año (consistentes con el dashboard, estables al agregar años)
PALETA_ANIOS = ["#27ae60", "#2980b9", "#e74c3c", "#8e44ad", "#f39c12",
                "#16a085", "#c0392b", "#2c3e50", "#d35400", "#7f8c8d",
                "#e67e22", "#1abc9c"]
COLOR_ANIO = {a: PALETA_ANIOS[i % len(PALETA_ANIOS)]
              for i, a in enumerate(range(2019, 2031))}


def grafica_meses(datos):
    """Eventos por mes: una línea por año + promedio mensual entre años."""
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    tmp = datos.dropna(subset=["fecha"]).copy()
    if len(tmp):
        tmp["anio"] = tmp["fecha"].dt.year.astype(int)
        serie = tmp.groupby(["anio", "mes"]).size().reset_index(name="eventos")
        for anio in sorted(serie["anio"].unique()):
            s = (serie[serie["anio"] == anio].set_index("mes")["eventos"]
                 .reindex(range(1, 13)).fillna(0))
            ax.plot(range(1, 13), s.values, marker="o", markersize=3,
                    color=COLOR_ANIO.get(int(anio), "#888"), label=str(anio))
        # Promedio entre años por mes
        prom = serie.groupby("mes")["eventos"].mean().reindex(range(1, 13)).fillna(0)
        ax.plot(range(1, 13), prom.values, linestyle="--", color="#000000",
                linewidth=1.3, label="Promedio")
        ax.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.9)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(list(MESES_ABREV.values()), fontsize=8)
    ax.set_ylabel("Eventos")
    ax.set_title("Eventos por mes (por año)", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_a_buffer(fig)


def grafica_diaria(datos):
    """Serie diaria de eventos + promedio movil de 7 dias."""
    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    tmp = datos.dropna(subset=["fecha"]).copy()
    if len(tmp):
        diaria = (tmp.assign(dia=tmp["fecha"].dt.floor("D"))
                  .groupby("dia").size())
        if len(diaria):
            idx = pd.date_range(diaria.index.min(), diaria.index.max(),
                                freq="D", tz="UTC")
            diaria = diaria.reindex(idx, fill_value=0)
            media7 = diaria.rolling(7, min_periods=1, center=True).mean()
            ax.plot(diaria.index, diaria.values, color="#bdc3c7", linewidth=0.7)
            ax.plot(media7.index, media7.values, color="#e74c3c", linewidth=1.6)
    ax.set_ylabel("Eventos")
    ax.set_title("Eventos por día (gris) y promedio móvil 7 días (rojo)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_a_buffer(fig)


def grafica_anios(resumen, valor="eventos"):
    """Barras por año (eventos o área), coloreadas con el color fijo de cada año."""
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    colores = [COLOR_ANIO.get(int(a), "#888") for a in resumen["anio"]]
    if valor == "eventos":
        ax.bar(resumen["anio"].astype(str), resumen["eventos"], color=colores)
        ax.set_ylabel("Eventos")
        ax.set_title("Eventos por año", fontsize=11, fontweight="bold")
        # Etiqueta de variacion %
        for i, (_, r) in enumerate(resumen.iterrows()):
            if pd.notna(r.get("var_pct")):
                col = "#c0392b" if r["var_pct"] >= 0 else "#27ae60"
                ax.text(i, r["eventos"], f"{r['var_pct']:+.0f}%", ha="center",
                        va="bottom", fontsize=8, fontweight="bold", color=col)
    else:
        ax.bar(resumen["anio"].astype(str), resumen["area"], color=colores)
        ax.set_ylabel("Área (ha)")
        ax.set_title("Área quemada por año", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_a_buffer(fig)


def torta_anios(resumen, valor="eventos"):
    """Torta (dona) de eventos o área por año, con color fijo por año."""
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    colores = [COLOR_ANIO.get(int(a), "#888") for a in resumen["anio"]]
    vals = resumen[valor].values
    etiquetas = resumen["anio"].astype(str).values
    if vals.sum() > 0:
        ax.pie(vals, labels=etiquetas, colors=colores, autopct="%1.0f%%",
               startangle=90, wedgeprops=dict(width=0.45), textprops={"fontsize": 8})
    titulo = "Eventos por año" if valor == "eventos" else "Área por año"
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    return _fig_a_buffer(fig)


def grafica_top(datos, campo, titulo, por_area=False):
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    if por_area:
        g = datos.groupby(campo)["area"].sum().sort_values(ascending=True).tail(10)
        xlabel = "Área (ha)"
    else:
        g = datos.groupby(campo).size().sort_values(ascending=True).tail(10)
        xlabel = "Eventos"
    ax.barh(range(len(g)), g.values, color="#c0392b" if por_area else "#e74c3c")
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.index, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_a_buffer(fig)


# ----------------------------------------------------------------------
# GENERADOR DEL PDF
# ----------------------------------------------------------------------
def generar_pdf(datos, nivel, titulo_zona, periodo, cats_presentes,
                gdf_deptos=None, gdf_limite=None, conteo_por_depto=None,
                ruta_logo=None):
    """
    Devuelve los bytes de un PDF con el reporte.
      datos          : DataFrame ya filtrado (nivel + periodo + confianza)
      nivel          : 'Nacional' | 'Departamental' | 'Municipal'
      titulo_zona    : texto del area (ej. 'Vista nacional', 'Arauca')
      periodo        : texto del rango de fechas
      cats_presentes : lista de categorias de confianza presentes
      gdf_deptos     : GeoDataFrame de departamentos (para mapa nacional)
      gdf_limite     : GeoDataFrame del contorno (depto o municipio)
      conteo_por_depto: dict {departamento: nro_eventos} (para mapa nacional)
      ruta_logo      : ruta opcional al logo institucional para la portada
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            leftMargin=1.8 * cm, rightMargin=1.8 * cm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=18,
                        textColor=colors.HexColor("#c0392b"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13,
                        textColor=colors.HexColor("#1f3b57"))
    normal = styles["Normal"]
    small = ParagraphStyle("small", parent=normal, fontSize=8,
                           textColor=colors.grey)

    story = []

    # --- Encabezado institucional (logo + titulo) ---
    if ruta_logo:
        try:
            from reportlab.platypus import Table as _T
            logo_img = Image(ruta_logo, width=3.2 * cm, height=1.6 * cm,
                             kind="proportional")
            titulo_par = Paragraph(
                "<b>Incendios Forestales - Colombia</b><br/>"
                "<font size=9 color='#555555'>Unidad Nacional para la Gestión "
                "del Riesgo de Desastres<br/>Subdirección para el Conocimiento "
                "del Riesgo</font>",
                ParagraphStyle("tit", parent=styles["Title"], fontSize=16,
                               textColor=colors.HexColor("#c0392b"), alignment=0))
            enc = _T([[logo_img, titulo_par]], colWidths=[3.6 * cm, 13 * cm])
            enc.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            story.append(enc)
        except Exception:
            story.append(Paragraph("Incendios Forestales - Colombia", h1))
    else:
        story.append(Paragraph("Incendios Forestales - Colombia", h1))
        story.append(Paragraph(
            "Unidad Nacional para la Gestión del Riesgo de Desastres · "
            "Subdirección para el Conocimiento del Riesgo", small))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Fuente de datos: WildFire Solution - Ororatech", small))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Ámbito:</b> {titulo_zona}", normal))
    story.append(Paragraph(f"<b>Periodo:</b> {periodo}", normal))
    story.append(Paragraph(
        f"<b>Generado:</b> {dt.datetime.now():%Y-%m-%d %H:%M}", normal))
    story.append(Spacer(1, 12))

    # --- KPIs ---
    n = len(datos)
    area_total = datos["area"].sum() if n else 0
    area_prom = datos["area"].mean() if n else 0
    n_alta = datos[datos["cat"].isin(["0.6", "0.8", "1.0"])].shape[0]
    pct_alta = (100 * n_alta / n) if n else 0
    kpi_data = [
        ["Eventos", "Área total (ha)", "Área prom. (ha)", "% conf. ≥0.6"],
        [f"{n:,}", f"{area_total:,.0f}", f"{area_prom:,.1f}", f"{pct_alta:.0f}%"],
    ]
    kpi_tbl = Table(kpi_data, colWidths=[4.2 * cm] * 4)
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, 1), 15),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f4f6f7")),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 14))

    # --- Mapa ---
    story.append(Paragraph("Mapa", h2))
    try:
        if nivel == "Nacional" and gdf_deptos is not None:
            img_buf = mapa_nacional(gdf_deptos, conteo_por_depto or {})
        else:
            img_buf = mapa_puntos(gdf_limite, datos.dropna(subset=["lat", "lon"]),
                                  f"Incendios · {titulo_zona}")
        story.append(Image(img_buf, width=13 * cm, height=12 * cm))
    except Exception as e:
        story.append(Paragraph(f"(No se pudo generar el mapa: {e})", small))
    story.append(PageBreak())

    # --- Graficas ---
    story.append(Paragraph("Análisis", h2))
    story.append(Spacer(1, 6))

    g_dist = grafica_distribucion(datos, cats_presentes)
    story.append(Image(g_dist, width=11 * cm, height=7 * cm))
    story.append(Spacer(1, 8))

    if datos["fecha"].notna().any():
        g_mes = grafica_meses(datos)
        story.append(Image(g_mes, width=11 * cm, height=7 * cm))
        story.append(Spacer(1, 8))
        # Serie diaria (a todo el ancho)
        g_dia = grafica_diaria(datos)
        story.append(Image(g_dia, width=16 * cm, height=4.6 * cm))
        story.append(Spacer(1, 8))

    # --- Analisis comparativo por año (tortas + barras de tendencia) ---
    serie_anual = (datos.dropna(subset=["fecha"])
                   .assign(anio=lambda d: d["fecha"].dt.year.astype(int)))
    if len(serie_anual):
        resumen = (serie_anual.groupby("anio")
                   .agg(eventos=("id", "size"), area=("area", "sum"))
                   .reset_index().sort_values("anio"))
        resumen["var_pct"] = resumen["eventos"].pct_change() * 100

        story.append(PageBreak())
        story.append(Paragraph("Análisis comparativo por año", h2))
        story.append(Spacer(1, 6))
        # Tortas lado a lado (eventos y area)
        t_ev = torta_anios(resumen, "eventos")
        t_ar = torta_anios(resumen, "area")
        fila_tortas = Table([[Image(t_ev, width=7 * cm, height=7 * cm),
                              Image(t_ar, width=7 * cm, height=7 * cm)]],
                            colWidths=[8 * cm, 8 * cm])
        fila_tortas.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(fila_tortas)
        story.append(Spacer(1, 8))
        # Barras: eventos por año (con variacion %) y area por año
        story.append(Image(grafica_anios(resumen, "eventos"),
                            width=11 * cm, height=7 * cm))
        story.append(Spacer(1, 8))
        story.append(Image(grafica_anios(resumen, "area"),
                            width=11 * cm, height=7 * cm))

    # --- Tops segun nivel (al final del analisis) ---
    story.append(PageBreak())
    if nivel == "Nacional":
        story.append(Image(grafica_top(datos, "departamento",
                     "Top 10 departamentos por eventos"), width=11 * cm, height=7.5 * cm))
        story.append(Spacer(1, 8))
        story.append(Image(grafica_top(datos, "departamento",
                     "Top 10 departamentos por área", por_area=True),
                     width=11 * cm, height=7.5 * cm))
    elif nivel == "Departamental":
        story.append(Image(grafica_top(datos, "municipio",
                     "Top 10 municipios por eventos"), width=11 * cm, height=7.5 * cm))
        story.append(Spacer(1, 8))
        story.append(Image(grafica_top(datos, "municipio",
                     "Top 10 municipios por área", por_area=True),
                     width=11 * cm, height=7.5 * cm))

    story.append(PageBreak())

    # --- Tabla de principales eventos (top 20 por area) ---
    story.append(Paragraph("Principales eventos (por área)", h2))
    story.append(Spacer(1, 6))
    cols = ["id", "departamento", "municipio", "area", "confidence",
            "num_fires", "oldest_acquisition"]
    cols = [c for c in cols if c in datos.columns]
    top = datos.sort_values("area", ascending=False).head(20)[cols]
    # Encabezados legibles
    encab = {"id": "ID", "departamento": "Departamento", "municipio": "Municipio",
             "area": "Área (ha)", "confidence": "Conf.", "num_fires": "N° focos",
             "oldest_acquisition": "Inicio"}
    filas = [[encab.get(c, c) for c in cols]]
    for _, r in top.iterrows():
        fila = []
        for c in cols:
            v = r[c]
            if c == "area":
                fila.append(f"{v:,.1f}")
            elif c == "oldest_acquisition":
                fila.append(str(v)[:10])
            else:
                fila.append(str(v))
        filas.append(fila)
    tbl = Table(filas, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f4f6f7")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 16))

    # --- Pie: descargo de responsabilidad, autoria y creditos ---
    just = ParagraphStyle("just", parent=normal, fontSize=8,
                          textColor=colors.HexColor("#444444"), alignment=4,
                          leading=10)
    pie = ParagraphStyle("pie", parent=normal, fontSize=8,
                         textColor=colors.HexColor("#444444"), leading=10)
    story.append(Paragraph("<b>Descargo de responsabilidad</b>", h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "La UNGRD, a través de la SCR, comparte la siguiente información, "
        "deslindándose de cualquier responsabilidad sobre el uso que se le dé. "
        "Los datos presentados han sido generados a partir de información extraída "
        "de la plataforma WFS de Ororatech. Este visor se suministra con el "
        "propósito de orientar la toma de decisiones; sin embargo, su correcto uso "
        "y aplicación son responsabilidad exclusiva de cada persona o entidad "
        "territorial.", just))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Realizado por:</b> Jorge Alpala "
        "(jorge.alpala@gestiondelriesgo.gov.co, UNGRD - SCR)<br/>"
        "<b>Elaboración:</b> Mayo de 2026", pie))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Fuentes y créditos:</b> Datos: OroraTech WildFire Solution (WFS) · "
        "Límites: DIVIPOLA - DANE. Desarrollado con Python, Streamlit, "
        "Folium/Leaflet, GeoPandas, Shapely, Pandas, Altair, Matplotlib y "
        "ReportLab. Mapas base: OpenStreetMap · CARTO.", pie))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
