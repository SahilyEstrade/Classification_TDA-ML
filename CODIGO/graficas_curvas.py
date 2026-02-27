#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 10:33:28 2026

@author: sahy
"""
# =============================================================================
# LIBRERIAS
# =============================================================================
import os
import pandas as pd
import numpy as np
import gudhi as gd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

##Script mios
import cfg as cfg
from organizacion_data import *
from curvas_persistencia import *

def matriz_betti(bettiData, dim, maxYlim):
    """
    Crea un mapa de calor que muestra cómo cambia el número de Betti
    (agujeros topológicos) a lo largo del tiempo y del parámetro epsilon.
    """

    # Filtrar por dimensión
    df = bettiData[bettiData["dimension"] == dim]

    # Extraer valores únicos ordenados
    t_vals = np.sort(df["t"].unique())
    eps_vals = np.sort(df["epsilon"].unique())

    # Crear matriz Betti (epsilon x tiempo)
    betti_matrix = np.full((len(eps_vals), len(t_vals)), np.nan)

    for i, eps in enumerate(eps_vals):
        for j, t in enumerate(t_vals):
            val = df[(df["epsilon"] == eps) & (df["t"] == t)]["betticount"]
            if not val.empty:
                betti_matrix[i, j] = val.values[0]

    # Crear figura
    plt.figure(figsize=(10, 6))

    # Mapa de calor
    im = plt.imshow(
        betti_matrix,
        aspect="auto",
        origin="lower",
        cmap="rainbow",
        extent=[t_vals.min(), t_vals.max(), eps_vals.min(), eps_vals.max()],
        vmin=0,
        vmax=np.nanmax(betti_matrix)
    )

    # Barra de color
    cbar = plt.colorbar(im)
    cbar.set_label("Betti count")

    # Límites y etiquetas
    plt.ylim(0, maxYlim)
    plt.xlabel("Simulation time t")
    plt.ylabel("Proximity Parameter ε")
    plt.title(f"Matriz de BettiCounts for Dimension {dim}")

    plt.tight_layout()
    plt.show()



def crockerplot_grafic(bettiData, dim, maxYlim):
    """
    Crea un CROCKER plot (Contour Realization Of Computed k-dimensional
    Hole Evolution in the Rips complex).
    
    Representa curvas de nivel del número de Betti en función del
    tiempo y del parámetro epsilon.
    """

    # Filtrar por dimensión
    df = bettiData[bettiData["dimension"] == dim]

    # Definir niveles de contorno (breaks)
    if dim == 0:
        breaks = np.arange(0, 61)
    else:
        breaks = np.arange(0, 21)

    # Extraer valores únicos ordenados
    t_vals = np.sort(df["t"].unique())
    eps_vals = np.sort(df["epsilon"].unique())

    # Construir la malla Z = betticount(epsilon, t)
    Z = np.full((len(eps_vals), len(t_vals)), np.nan)

    for i, eps in enumerate(eps_vals):
        for j, t in enumerate(t_vals):
            val = df[(df["epsilon"] == eps) & (df["t"] == t)]["betticount"]
            if not val.empty:
                Z[i, j] = val.values[0]

    # Crear malla para contornos
    T, E = np.meshgrid(t_vals, eps_vals)

    # Figura
    plt.figure(figsize=(10, 6))

    # Gráfico de contornos
    contour = plt.contour(
        T,
        E,
        Z,
        levels=breaks,
        cmap="rainbow",
        linewidths=1.2
    )

    # Barra de color
    cbar = plt.colorbar(contour)
    cbar.set_label("Betti count")

    # Límites y etiquetas
    plt.ylim(0, maxYlim)
    plt.xlabel("Simulation time t")
    plt.ylabel("Proximity Parameter ε")
    plt.title(f"Crocker plot for dimension {dim}")

    plt.tight_layout()
    plt.show()
    
    
#Gruardar matrices
def guardar_matrices(betti_0, betti_1, betti, betti_euler,
                     base_path, ev, i):

    rutas = {
        "b0": os.path.join(base_path, "b0"),
        "b1": os.path.join(base_path, "b1"),
        "b": os.path.join(base_path, "b"),
        "euler": os.path.join(base_path, "euler")
    }

    for r in rutas.values():
        os.makedirs(r, exist_ok=True)

    betti_0.to_csv(f"{rutas['b0']}/mb0_{ev}_{i}.csv", index=False)
    betti_1.to_csv(f"{rutas['b1']}/mb1_{ev}_{i}.csv", index=False)
    betti.to_csv(f"{rutas['b']}/mb_{ev}_{i}.csv", index=False)
    betti_euler.to_csv(f"{rutas['euler']}/mbe_{ev}_{i}.csv", index=False)



# =============================================================================


