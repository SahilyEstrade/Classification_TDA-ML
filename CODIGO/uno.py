#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 10:41:32 2026

@author: sahy

Organizacion de los datos para darle de entrada 

FUNCION BIEN 13 de enero!
"""
#------------------------------------------------------------------------------
# LIBRERIAS
#------------------------------------------------------------------------------
import os
from pathlib import Path
import pandas as pd
import re
import cfg as cfg

from dos import (
    creacion_df_total_simulaciones,
    individuos_cuadrado_df,
    getIntervals,
    turnIntervalsIntoGrid
)

#------------------------------------------------------------------------------
# PARÁMETROS GLOBALES
#------------------------------------------------------------------------------

# Marco del cuadrado de estudio
longX = 2
longY = 2

# Parámetros topológicos
L = 2.5
mymaxscale = 0.35
numepsilon = 50
mymaxdimension = 1

#------------------------------------------------------------------------------
# RUTAS
#------------------------------------------------------------------------------

try:
    root_dir = Path(__file__).resolve().parent
except NameError:
    root_dir = Path().resolve()

data_dir = root_dir.parent.parent / cfg.FOLDER_DATA_NAME
result_dir = root_dir.parent / "RESULTADOS"

#------------------------------------------------------------------------------
# FUNCIONES
#------------------------------------------------------------------------------

def compute_homology(data, salto):
    frames = data["Num_fotogramas"].unique()
    interval_time = frames[::salto]

    homologydata = []

    for time in interval_time:
        mydata = data.loc[
            data["Num_fotogramas"] == time, ["X", "Y"]
        ]

        newdata = getIntervals(
            mydata,
            L=L,
            mymaxdimension=mymaxdimension,
            mymaxscale=mymaxscale,
            thistime=time
        )

        homologydata.append(newdata)

    return pd.concat(homologydata, ignore_index=True)


def build_betti_matrices(homologydata):
    griddata = turnIntervalsIntoGrid(
        homologydata,
        numEpsilons=numepsilon,
        mymaxscale=mymaxscale
    )

    betti_0 = (
        griddata[griddata["dimension"] == 0]
        .drop(columns=["dimension"])
        .pivot(index="epsilon", columns="t", values="betticount")
        .fillna(0)
    )

    betti_1 = (
        griddata[griddata["dimension"] == 1]
        .drop(columns=["dimension"])
        .pivot(index="epsilon", columns="t", values="betticount")
        .fillna(0)
    )

    betti_euler = betti_0 - betti_1
    betti = pd.concat([betti_0, betti_1])

    return betti_0, betti_1, betti_euler, betti


def save_betti_matrices(betti_0, betti_1, betti_euler, betti, base_path, ev, idx=None):
    paths = {
        "b0": os.path.join(base_path, "b0"),
        "b1": os.path.join(base_path, "b1"),
        "be": os.path.join(base_path, "euler"),
        "b":  os.path.join(base_path, "b")
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    suffix = f"_{idx}" if idx is not None else ""

    betti_0.to_csv(f"{paths['b0']}/mb0_{ev}{suffix}.csv")
    betti_1.to_csv(f"{paths['b1']}/mb1_{ev}{suffix}.csv")
    betti_euler.to_csv(f"{paths['be']}/mbe_{ev}{suffix}.csv")
    betti.to_csv(f"{paths['b']}/mb_{ev}{suffix}.csv")



def creacion_df_simulacion(nombre_carpeta, base_path="/Users/sahy/Documents/trabajo Cristina/Am66"):
    """
    Lee todos los archivos de una carpeta de simulación, limpia los datos corruptos
    y devuelve un DataFrame con columnas: Num_fotogramas, X, Y, Individuo, Tiempo
    """
    
    import os
    import pandas as pd
    
    ruta = os.path.join(base_path, nombre_carpeta, "tracorrTh35")
    lista_archivos = sorted([os.path.join(ruta, f) for f in os.listdir(ruta)])

    df_lista = []

    for i, archivo in enumerate(lista_archivos, start=1):
        # BLOQUE FINAL DEFINITIVO
        df = pd.read_csv(
            archivo,
            sep=r"\s+",
            header=None,
            usecols=[0, 1, 2],
            names=["Num_fotogramas", "X", "Y"],
            engine="python",
            encoding="latin-1",
            encoding_errors="ignore",
            comment="#",
            on_bad_lines="skip"
        )

        # Forzar que los valores sean numéricos
        df["Num_fotogramas"] = pd.to_numeric(df["Num_fotogramas"], errors="coerce")
        df["X"] = pd.to_numeric(df["X"], errors="coerce")
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")

        # Eliminar filas corruptas
        df = df.dropna(subset=["Num_fotogramas", "X", "Y"])

        df["Individuo"] = f"{i:03d}"
        df["Tiempo"] = df["Num_fotogramas"] / 25.0

        df_lista.append(df)

    data_archivos = pd.concat(df_lista, ignore_index=True)
    data_archivos["Individuo"] = data_archivos["Individuo"].astype("category")

    # Aplicar filtros de tiempo inicial y final
    data_archivos = (
        data_archivos
        .pipe(quitar_tiempo_inicial)
        .pipe(quitar_tiempo_final)
        .sort_values(["Individuo", "Tiempo"])
    )

    return data_archivos



def quitar_tiempo_inicial(data_archivos):
    df_puerta = individuos_cuadrado_df(data_archivos, longX=1, longY=0.5)
    df_puerta = df_puerta.sort_values("Tiempo")

    df_puerta["cuadrado"] = np.where(
        (df_puerta["X"].abs() <= 2) & (df_puerta["Y"] <= 2),
        "Si", "No"
    )

    primer_tiempo = df_puerta["Tiempo"].iloc[0]

    individuos_iniciales = (
        df_puerta[
            (df_puerta["Tiempo"] == primer_tiempo) &
            (df_puerta["cuadrado"] == "Si")
        ]["Individuo"]
        .unique()
    )

    t_inicio = primer_tiempo

    for t in df_puerta["Tiempo"].unique():
        df_t = df_puerta[
            (df_puerta["Tiempo"] == t) &
            (df_puerta["Individuo"].isin(individuos_iniciales))
        ]

        if len(df_t) < len(individuos_iniciales):
            t_inicio = t
            break

    return data_archivos[data_archivos["Tiempo"] >= t_inicio]


def quitar_tiempo_final(data_archivos):
    conteo = (
        data_archivos
        .groupby("Tiempo")
        .size()
        .reset_index(name="Num_peatones")
    )

    umbral = conteo["Num_peatones"].max() * 0.2
    tiempos_validos = conteo[conteo["Num_peatones"] > umbral]["Tiempo"]

    return data_archivos[data_archivos["Tiempo"].isin(tiempos_validos)]


def creacion_df_total_simulaciones(lista_archivos, base_path):
    dfs = []

    for carpeta in lista_archivos:
        df = creacion_df_simulacion(
            nombre_carpeta=carpeta,
            base_path=base_path
        )
        df["id"] = carpeta[-6:]
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def individuos_cuadrado_df(data_archivos, longX, longY):
    return (
        data_archivos[
            (data_archivos["X"].abs() <= longX) &
            (data_archivos["Y"] <= longY)
        ]
        .copy()
    )



def datas_consecutivos(data, n_t):
    data = data.sort_values("Num_fotogramas")
    frames = data["Num_fotogramas"].unique()

    return [
        data[data["Num_fotogramas"].isin(frames[i:i + n_t])]
        for i in range(len(frames) - n_t + 1)
    ]


def getIntervals(mydata, L, mymaxdimension, mymaxscale, thistime):
    X = mydata[["X", "Y"]].values / L
    dist_matrix = squareform(pdist(X))

    rips = gudhi.RipsComplex(
        distance_matrix=dist_matrix,
        max_edge_length=mymaxscale
    )

    simplex_tree = rips.create_simplex_tree(
        max_dimension=mymaxdimension
    )
    simplex_tree.persistence()

    data = []

    for dim in range(mymaxdimension + 1):
        for birth, death in simplex_tree.persistence_intervals_in_dimension(dim):
            data.append([thistime, dim, birth, death])

    return pd.DataFrame(
        data,
        columns=["t", "dimension", "birth", "death"]
    )


def turnIntervalsIntoGrid(homologydata, numEpsilons, mymaxscale):
    epsilons = np.linspace(0, mymaxscale, numEpsilons)
    records = []

    for dim in homologydata["dimension"].unique():
        for t in homologydata["t"].unique():
            snap = homologydata[
                (homologydata["dimension"] == dim) &
                (homologydata["t"] == t)
            ]

            if snap.empty:
                continue

            births = snap["birth"].values
            deaths = snap["death"].values

            for e in epsilons:
                c = (births <= e).sum() - (deaths < e).sum()
                records.append([t, dim, e, c])

    return pd.DataFrame(
        records,
        columns=["t", "dimension", "epsilon", "betticount"]
    )

#------------------------------------------------------------------------------
# CARGA DE DATOS
#------------------------------------------------------------------------------

ruta = Path("/Users/sahy/Documents/trabajo Cristina/Am66")

lista_carpetas = [
    f.name for f in ruta.iterdir()
    if f.is_dir() and re.match(r"20170505_.*", f.name)
]

data_archivos_total = creacion_df_total_simulaciones(
    lista_carpetas,
    base_path=str(ruta)   # string, no Path
)

# Recorte espacial
data_archivos_total_delimitado = individuos_cuadrado_df(
    data_archivos_total, longX, longY
)

#-------ejecucion
cant_frames = 100
salto = 5

ids = data_sin_empujar["id"].unique()[:3]

process_evacuations(
    ids,
    data_archivos_total_delimitado,
    base_path="/Users/sahy/Documents/Peatones/Clasificador_Cristina/RESULTADOS/sin_empujar_100",
    cant_frames=cant_frames,
    salto=salto
)


ids = data_empujando["id"].unique()[:3]

process_evacuations(
    ids,
    data_archivos_total_delimitado,
    base_path="/Users/sahy/Documents/Peatones/Clasificador_Cristina/RESULTADOS/empujando_100",
    cant_frames=cant_frames,
    salto=salto
)


cant_frames = 1
salto = 1

ids = data_empujando["id"].unique()[:3]

process_evacuations(
    ids,
    data_archivos_total_delimitado,
    base_path= "/Users/sahy/Documents/Peatones/Clasificador_Cristina/RESULTADOS/empujando_frame",
    cant_frames=cant_frames,
    salto=salto
)



