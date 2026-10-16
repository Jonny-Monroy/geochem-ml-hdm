# -*- coding: utf-8 -*-
"""
HDM Geochemistry Explorer — Demo de portafolio
================================================
App interactiva de análisis geoquímico inspirada en la aplicación de producción
desarrollada para el proyecto *Huella Digital de Minerales* (Servicio Geológico
Colombiano). Esta versión demo corre 100% local sobre una muestra de datos
(SQLite) e implementa el mismo núcleo analítico:

  * Exploración con filtros en cascada (técnica, geografía, tipo de muestra)
  * Espectros químicos ordenados por número atómico
  * Matriz de correlación multielemento
  * PCA (componentes principales) con biplot de individuos
  * Clasificación supervisada: LDA y Random Forest (procedencia geográfica)
  * Mapa interactivo de muestras (Folium)
  * Tabla dinámica con descarga CSV

Ejecución:
    pip install -r requirements.txt
    shiny run app/app.py --reload
"""

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import sqlite3

import folium
from folium.plugins import MarkerCluster, Fullscreen

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_plotly

# ---------------------------------------------------------------------------
# Datos: la demo se sirve de un SQLite embebido (se descomprime al primer uso)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DB = HERE.parent / "data" / "hdm_demo.sqlite"
ZIP = HERE.parent / "data" / "hdm_demo.sqlite.zip"

if not DB.exists() and ZIP.exists():
    with zipfile.ZipFile(ZIP) as zf:
        zf.extractall(DB.parent)

COLUMNAS = [
    "codigo_muestra", "tipo_muestra", "subtipo_muestra", "fecha_muestreo",
    "wgs84_latitud", "wgs84_longitud", "proyecto", "laboratorio_nombre",
    "tecnica_analitica", "elemento", "unidad", "valor_original", "valor_ppm",
    "municipio", "departamento",
]

with sqlite3.connect(DB) as con:
    df_full = pd.read_sql(
        "SELECT {} FROM resultados".format(", ".join(COLUMNAS)), con
    )

# ---------------------------------------------------------------------------
# Número atómico para ordenar los espectros químicos
# ---------------------------------------------------------------------------
NUMERO_ATOMICO = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
}

TECNICAS_TOP = (
    df_full["tecnica_analitica"].value_counts().head(4).index.tolist()
)
ELEMENTOS_TOP = (
    df_full["elemento"].value_counts().head(20).index.tolist()
)
DEPARTAMENTOS = sorted(
    df_full.loc[df_full["departamento"].fillna("") != "", "departamento"]
    .unique()
    .tolist()
)
TIPOS_MUESTRA = sorted(
    df_full["tipo_muestra"].fillna("(sin tipo)").unique().tolist()
)

TECNICA_DEFAULT = next(
    (t for t in TECNICAS_TOP if "ICP-MS" in t), TECNICAS_TOP[0]
)

MIN_MUESTRAS_CLASE = 30  # clases con menos muestras se excluyen del modelo

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            body { font-family: 'Segoe UI', Roboto, sans-serif;
                   background: #f4f6f4; margin: 0; }
            .hdr { background: linear-gradient(90deg, #0E4D64, #1B9E77);
                   color: #fff; padding: 18px 28px; border-radius: 0 0 14px 14px; }
            .hdr h1 { margin: 0; font-size: 1.5rem; letter-spacing: .5px; }
            .hdr p  { margin: 4px 0 0; opacity: .85; font-size: .9rem; }
            .kpi-card { background: #fff; border-radius: 12px; padding: 14px 18px;
                        box-shadow: 0 1px 4px rgba(0,0,0,.12); text-align: center; }
            .kpi-card .v { font-size: 1.6rem; font-weight: 700; color: #0E4D64; }
            .kpi-card .l { font-size: .78rem; color: #556; text-transform: uppercase;
                           letter-spacing: .6px; }
            .tab-content { background: #fff; border-radius: 12px; padding: 18px;
                           box-shadow: 0 1px 4px rgba(0,0,0,.10); margin-top: 14px; }
        """)
    ),
    ui.tags.div(
        ui.tags.h1("HDM Geochemistry Explorer"),
        ui.tags.p(
            "Análisis geoquímico exploratorio + Machine Learning · "
            "demo derivada del proyecto Huella Digital de Minerales (SGC)"
        ),
        class_="hdr",
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_selectize(
                "tec", "Técnica analítica",
                choices=TECNICAS_TOP, selected=TECNICA_DEFAULT,
            ),
            ui.input_selectize(
                "deptos", "Departamento(s)",
                choices=DEPARTAMENTOS, selected=DEPARTAMENTOS, multiple=True,
            ),
            ui.input_selectize(
                "tipos", "Tipo de muestra",
                choices=TIPOS_MUESTRA, selected=TIPOS_MUESTRA[:1], multiple=True,
            ),
            ui.input_selectize(
                "elems", "Elementos (espectro/ML)",
                choices=ELEMENTOS_TOP,
                selected=["Au", "Ag", "As", "Cu", "Zn", "Pb", "Mn", "Fe",
                          "Ni", "Co", "Cr", "Ba", "Sr", "La", "Th", "U"],
                multiple=True,
            ),
            ui.input_action_button(
                "analizar", "ANALIZAR", class_="btn btn-success w-100"
            ),
            width=290,
        ),
        ui.navset_tab(
            ui.nav_panel("Resumen", ui.output_ui("kpis")),
            ui.nav_panel(
                "Operativo",
                ui.output_ui("op_kpis"),
                ui.output_ui("op_nota"),
                output_widget("op_treemap"),
                output_widget("op_barras"),
            ),
            ui.nav_panel("Mapa", ui.output_ui("mapa")),
            ui.nav_panel(
                "Espectro",
                output_widget("espectro"),
                ui.output_ui("nota_espectro"),
            ),
            ui.nav_panel("Correlación", output_widget("correlacion")),
            ui.nav_panel("PCA", output_widget("pca_scatter")),
            ui.nav_panel(
                "Machine Learning",
                ui.output_plot("ml_fig", height="560px"),
                ui.output_ui("ml_resumen"),
            ),
            ui.nav_panel(
                "Tabla",
                ui.output_data_frame("tabla"),
                ui.download_button("descarga_csv", "Descargar CSV",
                                   class_="btn btn-outline-success mt-2"),
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def server(input, output, session):

    @reactive.Calc
    def datos_filtrados():
        """Filtra el DataFrame largo según los selects del panel lateral."""
        df = df_full
        tec = input.tec()
        if tec:
            df = df[df["tecnica_analitica"] == tec]
        sel_d = [x for x in input.deptos() if x]
        if sel_d:
            df = df[df["departamento"].isin(sel_d)]
        sel_t = [x for x in input.tipos() if x]
        if sel_t:
            df = df[df["tipo_muestra"].isin(sel_t)]
        return df

    @reactive.Calc
    @reactive.event(input.analizar, ignore_none=False)
    def resultado():
        """Dataset largo filtrado + matriz ancha (muestras × elementos)."""
        d = datos_filtrados()
        elems = [e for e in input.elems() if e in set(d["elemento"])]
        d = d[d["elemento"].isin(elems)]
        if len(d) == 0:
            return {"largo": d, "ancho": pd.DataFrame(), "elems": []}

        ancho_val = d.pivot_table(
            index="codigo_muestra", columns="elemento",
            values="valor_ppm", aggfunc="median",
        )
        meta = (
            d.drop_duplicates("codigo_muestra")
            .set_index("codigo_muestra")[
                ["departamento", "municipio", "tipo_muestra",
                 "wgs84_latitud", "wgs84_longitud"]
            ]
        )
        ancho = ancho_val.join(meta, how="inner")
        return {"largo": d, "ancho": ancho, "elems": elems}

    # ------------------------------------------------------------- Resumen
    @render.ui
    def kpis():
        r = resultado()
        largo, ancho = r["largo"], r["ancho"]
        n_muestras = largo["codigo_muestra"].nunique()
        n_deptos = largo["departamento"].nunique()
        elems_txt = ", ".join(r["elems"][:12]) + (
            "…" if len(r["elems"]) > 12 else ""
        )
        cards = (
            ("Muestras", f"{n_muestras:,}"),
            ("Registros", f"{len(largo):,}"),
            ("Departamentos", str(n_deptos)),
            ("Elementos", str(len(r["elems"]))),
        )
        html = "".join(
            f"<div class='kpi-card'><div class='v'>{v}</div>"
            f"<div class='l'>{l}</div></div>"
            for l, v in cards
        )
        return ui.tags.div(
            ui.HTML(
                "<div style='display:grid;gap:14px;"
                "grid-template-columns:repeat(auto-fit,minmax(160px,1fr));'>"
                f"{html}</div>"
            ),
            ui.tags.p(
                ui.tags.b("Elementos activos: "), elems_txt,
                style_="margin-top:14px;color:#334;",
            ),
        )

    # ------------------------------------------------------------- Operativo
    @render.ui
    def op_kpis():
        largo = datos_filtrados()
        if largo.empty:
            return ui.p("Sin datos para los filtros seleccionados.")
        base = largo.drop_duplicates("codigo_muestra")
        cards = (
            ("Muestras", f"{len(base):,}".replace(",", ".")),
            ("Departamentos", str(base["departamento"].nunique())),
            ("Municipios", str(base["municipio"].nunique())),
            ("Laboratorios", str(base["laboratorio_nombre"].nunique())),
            ("Proyectos", str(base["proyecto"].nunique())),
        )
        html = "".join(
            f"<div class='kpi-card'><div class='v'>{v}</div>"
            f"<div class='l'>{l}</div></div>"
            for l, v in cards
        )
        return ui.HTML(
            "<div style='display:grid;gap:14px;"
            "grid-template-columns:repeat(auto-fit,minmax(150px,1fr));'>"
            f"{html}</div>"
        )

    @render.ui
    def op_nota(self=None):
        return ui.tags.p(
            "Réplica del tablero operativo productivo: en el SGC mantengo 4 "
            "dashboards de este estilo (Depósitos, Estaciones, Muestras y "
            "Pozos) con KPIs, % de aprobación, treemaps de laboratorios y "
            "mapas, consumiendo las tablas hdm_tablero_* de PostgreSQL.",
            style_="margin-top:12px;color:#556;font-size:.88rem;",
        )

    @render_plotly
    def op_treemap():
        largo = datos_filtrados()
        if largo.empty:
            return go.Figure()
        base = largo.drop_duplicates("codigo_muestra")
        fig = px.treemap(
            base,
            path=[px.Constant("Laboratorios"),
                  base["laboratorio_nombre"].fillna("(sin laboratorio)"),
                  base["tipo_muestra"].fillna("(sin tipo)")],
            title="Muestras por laboratorio y tipo",
        )
        fig.update_layout(height=420)
        return fig

    @render_plotly
    def op_barras():
        largo = datos_filtrados()
        if largo.empty:
            return go.Figure()
        base = largo.drop_duplicates("codigo_muestra")
        conteo = (
            base["tipo_muestra"].fillna("(sin tipo)")
            .value_counts().reset_index()
        )
        conteo.columns = ["tipo_muestra", "n"]
        conteo = conteo.sort_values("n")
        fig = px.bar(
            conteo, y="tipo_muestra", x="n", orientation="h",
            color="tipo_muestra",
            color_discrete_sequence=px.colors.qualitative.Dark24,
            labels={"n": "Muestras"},
            title="Muestras por tipo",
        )
        fig.update_layout(
            height=max(320, 34 * len(conteo)),
            showlegend=False,
        )
        return fig

    # ---------------------------------------------------------------- Mapa
    @render.ui
    def mapa():
        ancho = resultado()["ancho"]
        if ancho.empty:
            return ui.p("Sin datos para los filtros seleccionados.")
        pts = ancho.dropna(subset=["wgs84_latitud", "wgs84_longitud"])
        if pts.empty:
            return ui.p("No hay coordenadas disponibles.")
        pts = pts.sample(n=min(len(pts), 3000), random_state=42)

        centro = [pts["wgs84_latitud"].mean(), pts["wgs84_longitud"].mean()]
        m = folium.Map(location=centro, zoom_start=6, tiles="OpenStreetMap")
        cluster = MarkerCluster().add_to(m)
        colores_deptos = {
            d: c for d, c in zip(
                pts["departamento"].dropna().unique(),
                px.colors.qualitative.Dark24,
            )
        }
        for _, row in pts.iterrows():
            dep = row.get("departamento") or "?"
            folium.Marker(
                location=[row["wgs84_latitud"], row["wgs84_longitud"]],
                popup=(f"<b>{row.name}</b><br>{dep}<br>"
                       f"{row.get('municipio') or ''}"),
            ).add_to(cluster)
        Fullscreen().add_to(m)
        return ui.HTML(m._repr_html_())

    # ------------------------------------------------------------- Espectro
    @render_plotly
    def espectro():
        r = resultado()
        largo, elems = r["largo"], r["elems"]
        if largo.empty:
            return go.Figure()
        esp = (
            largo.assign(v=np.log1p(largo["valor_ppm"].clip(lower=0)))
            .groupby(["departamento", "elemento"], as_index=False)["v"]
            .mean()
        )
        esp["z"] = esp["elemento"].map(NUMERO_ATOMICO)
        esp = esp[esp["z"].notna()].sort_values(["departamento", "z"])
        esp["elemento"] = (
            esp["elemento"] + " (Z=" + esp["z"].astype(int).astype(str) + ")"
        )
        orden = (
            esp.drop_duplicates("elemento")
            .sort_values("z")["elemento"]
            .tolist()
        )
        fig = px.line(
            esp, x="elemento", y="v", color="departamento", markers=True,
            labels={"v": "log(1 + valor ppm) — media"},
            title="Espectro químico medio por departamento (ordenado por Z)",
        )
        fig.update_xaxes(categoryorder="array", categoryarray=orden)
        fig.update_layout(template="plotly_white", height=520)
        return fig

    @render.ui
    def nota_espectro(self=None):
        return ui.tags.p(
            "Firmas espectrales: cada traza resume la composición media de un "
            "departamento. Diferencias sistemáticas en grupos de elementos "
            "(p. ej. Au-As-Sb vs. Cu-Pb-Zn) sustentan la clasificación de "
            "procedencia con modelos supervisados.",
            style_="color:#556;font-size:.88rem;",
        )

    # -------------------------------------------------------- Correlación
    @render_plotly
    def correlacion():
        ancho, elems = resultado()["ancho"], resultado()["elems"]
        cols = [c for c in elems if c in ancho.columns]
        if not cols:
            return go.Figure()
        corr = ancho[cols].apply(np.log1p).corr(numeric_only=True)
        fig = px.imshow(
            corr, text_auto=".2f", zmin=-1, zmax=1,
            color_continuous_scale="RdBu", aspect="auto",
            title="Correlación entre elementos (escala log)",
        )
        fig.update_layout(height=560)
        return fig

    # ----------------------------------------------------------------- PCA
    @render_plotly
    def pca_scatter():
        ancho = resultado()["ancho"]
        df_pca, varianza, clases = _pca_datos(ancho)
        if df_pca is None:
            return go.Figure()
        fig = px.scatter(
            df_pca, x="PC1", y="PC2", color="departamento",
            hover_name=df_pca.index, opacity=.75,
            title="PCA — biplot de individuos (escala log + estandarizada)",
            labels={"PC1": f"PC1 ({varianza[0]:.1%})",
                    "PC2": f"PC2 ({varianza[1]:.1%})"},
        )
        fig.update_layout(template="plotly_white", height=560)
        return fig

    # ----------------------------------------------------------------- ML
    @render.plot
    def ml_fig():
        ancho = resultado()["ancho"]
        datos_ml = _ml_datos(ancho)
        if datos_ml is None:
            fig, ax = plt.subplots()
            ax.text(.5, .5, "Datos insuficientes para entrenar (≥30 "
                            "muestras por clase)", ha="center", va="center")
            ax.axis("off")
            return fig
        X_train, X_test, y_train, y_test, X_all, y_all, elems, clases = datos_ml
        lda = LinearDiscriminantAnalysis(n_components=2)
        rf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42,
            n_jobs=-1,
        )
        lda.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        acc_lda = accuracy_score(y_test, lda.predict(X_test))
        acc_rf = accuracy_score(y_test, rf.predict(X_test))

        proy = lda.transform(X_all)
        importancias = pd.Series(rf.feature_importances_, index=elems)
        top = importancias.sort_values(ascending=False).head(15)[::-1]
        cm = confusion_matrix(y_test, rf.predict(X_test), labels=clases)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        for i, dep in enumerate(clases):
            m = y_all == dep
            axes[0].scatter(proy[m, 0], proy[m, 1], s=10, alpha=.6,
                            label=f"{dep} (n={m.sum()})")
        axes[0].set_title(f"Proyección LDA · acc {acc_lda:.1%}")
        axes[0].set_xlabel("LD1"); axes[0].set_ylabel("LD2")
        axes[0].legend(fontsize=7)

        axes[1].barh(top.index, top.values, color="#1B9E77")
        axes[1].set_title(f"Random Forest · top-15 variables · acc {acc_rf:.1%}")
        axes[1].set_xlabel("Importancia")

        im = axes[2].imshow(cm, cmap="Greens")
        axes[2].set_xticks(range(len(clases)), clases, rotation=45,
                           ha="right", fontsize=7)
        axes[2].set_yticks(range(len(clases)), clases, fontsize=7)
        axes[2].set_title("Matriz de confusión RF (test)")
        for a in range(len(clases)):
            for b in range(len(clases)):
                axes[2].text(b, a, f"{cm[a, b]:,}", ha="center",
                             va="center", fontsize=7)
        plt.tight_layout()
        return fig

    @render.ui
    def ml_resumen(self=None):
        datos_ml = _ml_datos(resultado()["ancho"])
        if datos_ml is None:
            return None
        _, _, _, _, _, _, elems, clases = datos_ml
        return ui.tags.div(
            ui.tags.p(
                ui.tags.b("Configuración: "),
                f"split 70/30 estratificado · imputación por mediana · "
                f"log(1+x) + StandardScaler · {len(elems)} predictores · "
                f"{len(clases)} clases: {', '.join(clases)}",
            ),
            ui.tags.p(
                "En producción, el mismo esquema clasifica la procedencia de "
                "muestras auríferas con una precisión del 92% "
                "(validación cruzada).",
                style_="font-size:.85rem;color:#556;",
            ),
        )

    # ---------------------------------------------------------------- Tabla
    @render.data_frame
    def tabla():
        ancho = resultado()["ancho"].copy()
        if ancho.empty:
            return render.DataGrid(pd.DataFrame(), width="100%")
        ancho = ancho.round(3).reset_index()
        return render.DataGrid(
            ancho.head(2000), filters=True, width="100%", height=460
        )

    @render.download(filename="hdm_resultados.csv")
    def descarga_csv():
        ancho = resultado()["ancho"].copy()
        yield ancho.round(4).to_csv(index=True).encode("utf-8-sig")


# --------------------------------------------------------------- utilidades
def _matriz_ml(ancho):
    """Matriz X (log1p + imputación + escalado) e y (departamento)."""
    if ancho.empty:
        return None
    elems = [c for c in ancho.columns
             if c not in {"departamento", "municipio", "tipo_muestra",
                          "wgs84_latitud", "wgs84_longitud"}]
    df = ancho.dropna(subset=["departamento"]).copy()
    conteo = df["departamento"].value_counts()
    clases = conteo[conteo >= MIN_MUESTRAS_CLASE].index.tolist()
    df = df[df["departamento"].isin(clases)]
    if len(clases) < 2 or len(df) < 60:
        return None
    X = df[elems].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(np.log1p(X.clip(lower=0))),
        columns=elems, index=df.index,
    )
    return X, df["departamento"], elems, clases


def _ml_datos(ancho):
    base = _matriz_ml(ancho)
    if base is None:
        return None
    X, y, elems, clases = base
    Xs = pd.DataFrame(
        StandardScaler().fit_transform(X), columns=elems, index=X.index
    )
    idx = train_test_split(
        Xs.index, test_size=.3, stratify=y, random_state=42
    )
    tr, te = idx
    return (Xs.loc[tr], Xs.loc[te], y.loc[tr], y.loc[te], Xs, y, elems, clases)


def _pca_datos(ancho):
    base = _matriz_ml(ancho)
    if base is None:
        return None, None, None
    X, y, elems, clases = base
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(Xs)
    df_pca = pd.DataFrame(
        comps, columns=["PC1", "PC2"], index=X.index
    )
    df_pca["departamento"] = y.values
    return df_pca, pca.explained_variance_ratio_, clases


app = App(app_ui, server)
