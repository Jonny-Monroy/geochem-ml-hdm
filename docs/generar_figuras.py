# -*- coding: utf-8 -*-
"""
Genera las figuras del README a partir de la base de datos demo.

Uso:
    python docs/generar_figuras.py

Salida: docs/img/*.png
"""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from app import NUMERO_ATOMICO, MIN_MUESTRAS_CLASE  # noqa: E402

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "data" / "hdm_demo.sqlite"
IMG = HERE / "img"
IMG.mkdir(exist_ok=True)

ELEMS = ["Au", "Ag", "As", "Cu", "Zn", "Pb", "Mn", "Fe", "Ni", "Co",
         "Cr", "Ba", "Sr", "La", "Th", "U"]

with sqlite3.connect(DB) as con:
    df = pd.read_sql(
        "SELECT codigo_muestra, tecnica_analitica, elemento, valor_ppm, "
        "departamento, municipio, wgs84_latitud, wgs84_longitud, tipo_muestra "
        "FROM resultados",
        con,
    )

# Fijar la técnica multielemento dominante (mismo criterio de la app)
tec = df["tecnica_analitica"].value_counts().idxmax()
if "ICP-MS" not in tec:
    tec = next(t for t in df["tecnica_analitica"].unique() if "ICP-MS" in t)
d = df[(df["tecnica_analitica"] == tec) & (df["elemento"].isin(ELEMS))]
print("Técnica:", tec[:60], "| filas:", f"{len(d):,}")

ancho = d.pivot_table(index="codigo_muestra", columns="elemento",
                      values="valor_ppm", aggfunc="median")
meta = (d.drop_duplicates("codigo_muestra")
          .set_index("codigo_muestra")[
              ["departamento", "municipio", "tipo_muestra",
               "wgs84_latitud", "wgs84_longitud"]])
ancho = ancho.join(meta, how="inner")

# ---------------------------------------------------------------- Figura 1
# Espectro químico medio por departamento
esp = (d.assign(v=np.log1p(d["valor_ppm"].clip(lower=0)))
         .groupby(["departamento", "elemento"], as_index=False)["v"].mean())
esp["z"] = esp["elemento"].map(NUMERO_ATOMICO)
esp = esp[esp["z"].notna()].sort_values("z")
fig, ax = plt.subplots(figsize=(12, 5.2))
for dep, g in esp.groupby("departamento"):
    ax.plot(g["z"], g["v"], marker="o", ms=4, lw=1.4, label=f"{dep} (n={len(g)})")
for z, el in esp.drop_duplicates("z")[["z", "elemento"]].values:
    ax.annotate(el, (z, esp["v"].max()), fontsize=8, rotation=90,
               ha="center", va="top", color="#888")
ax.set_xlabel("Número atómico (Z)")
ax.set_ylabel("log(1 + valor ppm) — media")
ax.set_title("Firma espectral media por departamento — ICP-MS")
ax.legend(fontsize=8, ncol=3)
ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(IMG / "espectro_departamentos.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- Figura 2
# Mapa de muestras
pts = ancho.dropna(subset=["wgs84_latitud", "wgs84_longitud"])
fig, ax = plt.subplots(figsize=(9, 7.5))
for dep, g in pts.groupby("departamento"):
    ax.scatter(g["wgs84_longitud"], g["wgs84_latitud"], s=8, alpha=.6, label=dep)
ax.set_xlabel("Longitud WGS84"); ax.set_ylabel("Latitud WGS84")
ax.set_title(f"Distribución geográfica de {len(pts):,} muestras geoquímicas")
ax.legend(fontsize=8)
ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(IMG / "mapa_muestras.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- Figura 3
# Preparación de la matriz para PCA / modelos
elems = [c for c in ancho.columns
         if c not in {"departamento", "municipio", "tipo_muestra",
                      "wgs84_latitud", "wgs84_longitud"}]
dfm = ancho.dropna(subset=["departamento"]).copy()
conteo = dfm["departamento"].value_counts()
clases = conteo[conteo >= MIN_MUESTRAS_CLASE].index.tolist()
dfm = dfm[dfm["departamento"].isin(clases)]
X = dfm[elems].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
X = pd.DataFrame(SimpleImputer(strategy="median")
                 .fit_transform(np.log1p(X.clip(lower=0))),
                 columns=elems, index=dfm.index)
Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=elems, index=X.index)
y = dfm["departamento"]

pca = PCA(n_components=2, random_state=42)
comps = pca.fit_transform(Xs)
fig, ax = plt.subplots(figsize=(9.5, 6.5))
for dep in clases:
    m = (y == dep).values
    ax.scatter(comps[m, 0], comps[m, 1], s=10, alpha=.65, label=dep)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("PCA — separación de firmas geoquímicas por procedencia")
ax.legend(fontsize=8)
ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(IMG / "pca_biplot.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- Figura 4
# LDA + Random Forest
itr, ite = train_test_split(Xs.index, test_size=.3, stratify=y, random_state=42)
Xtr, Xte, ytr, yte = Xs.loc[itr], Xs.loc[ite], y.loc[itr], y.loc[ite]
lda = LinearDiscriminantAnalysis(n_components=2).fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                             random_state=42, n_jobs=-1).fit(Xtr, ytr)
acc_lda = accuracy_score(yte, lda.predict(Xte))
acc_rf = accuracy_score(yte, rf.predict(Xte))
print(f"Accuracy LDA: {acc_lda:.1%} | Accuracy RF: {acc_rf:.1%}")

proy = lda.transform(Xs)
imp = pd.Series(rf.feature_importances_, index=elems).sort_values().tail(15)
cm = confusion_matrix(yte, rf.predict(Xte), labels=clases)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for dep in clases:
    m = (y == dep).values
    axes[0].scatter(proy[m, 0], proy[m, 1], s=8, alpha=.6, label=dep)
axes[0].set_title(f"Proyección LDA · test acc {acc_lda:.1%}")
axes[0].legend(fontsize=7)
axes[1].barh(imp.index, imp.values, color="#1B9E77")
axes[1].set_title(f"Random Forest · top variables · test acc {acc_rf:.1%}")
axes[2].imshow(cm, cmap="Greens")
axes[2].set_xticks(range(len(clases)), clases, rotation=45, ha="right", fontsize=7)
axes[2].set_yticks(range(len(clases)), clases, fontsize=7)
axes[2].set_title("Matriz de confusión RF (test)")
for a in range(len(clases)):
    for b in range(len(clases)):
        axes[2].text(b, a, f"{cm[a, b]:,}", ha="center", va="center", fontsize=7)
plt.tight_layout()
plt.savefig(IMG / "ml_resultados.png", dpi=140)
plt.close()

print("Figuras generadas en", IMG)
