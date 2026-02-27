#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 09:30:09 2026

@author: sahy

Para calcular los crocker plot y betti
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


# =============================================================================
# HOMOLOGÍA PERSISTENTE
# =============================================================================

def getIntervals(mydata, thistime, L=None, mymaxdimension=None, mymaxscale=None):
    """"
    Calcula los intervalos de persistencia (diagramas de Vietoris–Rips)
    para un conjunto de puntos en un instante de tiempo dado.

    A partir de las posiciones de los individuos, se construye una matriz
    de distancias, se genera el complejo de Vietoris–Rips y se calculan
    los intervalos de persistencia para las distintas dimensiones
    topológicas.

    Parameters
    ----------
    mydata : pandas.DataFrame
        DataFrame con las posiciones de los individuos en un instante
        temporal específico.
    thistime : float
        Tiempo asociado al conjunto de datos.
    L : float, optional
        Factor de normalización espacial. Si no se especifica, se utiliza
        el valor definido en la configuración global (cfg.L).
    mymaxdimension : int, optional
        Dimensión máxima de los complejos simpliciales a considerar.
        Por defecto se toma de cfg.MYMAXDIMENSION.
    mymaxscale : float, optional
        Escala máxima para la construcción del complejo de Vietoris–Rips.
        Por defecto se toma de cfg.MYMAXSCALE.

    Returns
    -------
    list of dict
        Lista de diccionarios con los intervalos de persistencia, donde
        cada diccionario contiene:
        - t: tiempo del experimento
        - dimension: dimensión topológica
        - birth: tiempo de nacimiento
        - death: tiempo de muerte
    """     
    
    #Parámetros por defecto desde la configuración global
    if L is None:
       L = cfg.L
    if mymaxdimension is None:
        mymaxdimension = cfg.MYMAXDIMENSION
    if mymaxscale is None:
        mymaxscale = cfg.MYMAXSCALE

    # Si no hay suficientes puntos, no se puede construir el complejo
    if mydata.empty or len(mydata) < 2:
        return []
    
    # Coordenadas normalizadas de los puntos
    pts = mydata[[cfg.X_COL, cfg.Y_COL]].values / L

    # Matriz de distancias euclídeas entre todos los puntos
    dist_matrix = squareform(pdist(pts))
    
    # Construcción del complejo de Vietoris–Rips
    rips = gd.RipsComplex(
        distance_matrix=dist_matrix,
        max_edge_length=mymaxscale
    )
    
    # Creación del árbol simplicial hasta la dimensión máxima
    simplex_tree = rips.create_simplex_tree(
        max_dimension=mymaxdimension
    )

    # Cálculo de la persistencia
    persistence = simplex_tree.persistence()
    
    # Formateo de los intervalos de persistencia
    return [
        {
            "t": thistime,
            "dimension": dim,
            "birth": birth,
            "death": death
        }
        for dim, (birth, death) in persistence
        if not np.isnan(birth)
    ]


# def turnIntervalsIntoGrid(homologydata, numEpsilons = None, mymaxscale = None):
#     """
#     Transforma intervalos de persistencia en una grilla de números de Betti
#     evaluados en distintas escalas (epsilon) y tiempos.

#     Para cada dimensión topológica y cada instante temporal, la función
#     cuenta cuántos intervalos están activos en una escala epsilon dada,
#     construyendo así una representación tipo Betti curve.

#     Parameters
#     ----------
#     homologydata : pandas.DataFrame
#         DataFrame con los intervalos de persistencia. Debe contener las
#         columnas: 't', 'dimension', 'birth' y 'death'.
#     numEpsilons : int, optional
#         Número de valores de epsilon utilizados para discretizar la escala.
#         Por defecto se toma de cfg.NUMEPSILON.
#     mymaxscale : float, optional
#         Escala máxima considerada para epsilon. Por defecto se toma de
#         cfg.MYMAXSCALE.

#     Returns
#     -------
#     pandas.DataFrame
#         DataFrame con las columnas:
#         - t: tiempo
#         - dimension: dimensión topológica
#         - epsilon: escala
#         - betticount: número de Betti en esa escala
#     """

#     # Parámetros por defecto
#     if numEpsilons is None:
#        numEpsilons =cfg.NUMEPSILON
#     if mymaxscale is None:
#        mymaxscale = cfg.MYMAXSCALE
       
#        if homologydata.empty:
#            raise ValueError(
#             "homologydata está vacío: no se pudieron calcular intervalos"
#     )
    
#     dimensions = homologydata["dimension"].unique()
#     times = homologydata["t"].unique()

#     epsilons = np.linspace(0, mymaxscale, numEpsilons)

#     betti_rows = []

#     for dim in dimensions:
#         for t in times:
#             snapshot = homologydata[
#                 (homologydata["dimension"] == dim) &
#                 (homologydata["t"] == t)
#             ]

#             if snapshot.empty:
#                 continue

#             births = snapshot["birth"].values
#             deaths = snapshot["death"].values

#             for eps in epsilons:
#                 count = np.sum((births <= eps) & (deaths >= eps))
#                 betti_rows.append([t, dim, eps, count])

#     return pd.DataFrame(
#         betti_rows,
#         columns=["t", "dimension", "epsilon", "betticount"]
#     )
# =============================================================================
def turnIntervalsIntoGrid(
    homologydata,
    curve_type=None,
    dimension = None,
    numEpsilons=None,
    mymaxscale=None,
    **curve_kwargs
):
    """
    Construye una grilla (t, dimension, epsilon) usando la curva
    seleccionada (betti, entropy o euler).
    """

    if homologydata.empty:
        raise ValueError(
            "homologydata está vacío: no se pudieron calcular intervalos"
        )

    if numEpsilons is None:
        numEpsilons = cfg.NUMEPSILON

    if mymaxscale is None:
        mymaxscale = cfg.MYMAXSCALE
    
    if curve_type is None:
        curve_type = cfg.CURVA_DE_PERSISTENCIA
        
    if dimension is None:
        dimension = cfg.DIMENSION

    dimensions = np.sort(homologydata["dimension"].unique())
    times = np.sort(homologydata["t"].unique())

    epsilons = np.linspace(0, mymaxscale, numEpsilons)

    rows = []

    for t in times:

        # snapshot en tiempo externo t
        snapshot_t = homologydata[
            homologydata["t"] == t
        ]

        if snapshot_t.empty:
            continue

        # -----------------------------
        # Curvas que dependen de dimensión
        # -----------------------------
        if curve_type in ("betti", "entropy"):

            for dim in dimensions:

                df_dim = snapshot_t[
                    snapshot_t["dimension"] == dim
                ]

                if df_dim.empty:
                    continue

                eps, values = apply_curve(
                    intervals=df_dim,
                    curve_type=curve_type,
                    dimension=dim,
                    times=epsilons,
                    **curve_kwargs
                )

                for e, v in zip(eps, values):
                    rows.append([t, dim, e, v])

        # -----------------------------
        # Euler (no depende de dimensión)
        # -----------------------------
        else:  # euler

            eps, values = apply_curve(
                intervals=snapshot_t,
                curve_type="euler",
                times=epsilons
            )

            for e, v in zip(eps, values):
                # dimension = -1 para indicar global
                rows.append([t, -1, e, v])

    return pd.DataFrame(
        rows,
        columns=["t", "dimension", "epsilon", "value"]
    )

# =============================================================================
def crear_matrices(griddata, dim):

    matrix = (
        griddata[griddata["dimension"] == dim]
        .drop(columns=["dimension"])
        .pivot(index="epsilon", columns="t", values="value")
        .fillna(0)
    )
    return matrix




