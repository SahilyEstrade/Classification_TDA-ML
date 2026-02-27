#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 09:24:11 2026

@author: sahy
"""

import numpy as np
import cfg as cfg

# =============================================================================
# CURVAS DE PERSISTENCIA
# =============================================================================
#Partimos de que tenemos algo asi t | dimension | birth | death


def apply_curve(intervals, curve_type=None , dimension=None, times=None, **kwargs):
    """
    Aplica una de las curvas disponibles (betti, entropy o euler)
    según la selección del usuario.

    Parameters
    ----------
    intervals : pandas.DataFrame
        DataFrame con los intervalos de persistencia.

    curve_type : str
        Tipo de curva a calcular.
        Valores permitidos:
        - "betti"
        - "entropy"
        - "euler"
        - "betti_combined"

    dimension : int or None, optional
        Dimensión homológica requerida para las curvas:
        - betti
        - entropy

        Para la curva de Euler este parámetro se ignora.

    times : array-like or None
        Vector de tiempos en los que se evalúa la curva.

    **kwargs :
        Parámetros adicionales que se pasan a la función seleccionada
        (por ejemplo epsilon para entropy_curve).

    Returns
    -------
    times : numpy.ndarray
        Vector de tiempos.

    values : numpy.ndarray
        Valores de la curva seleccionada.
    """

    curve_type = curve_type.lower()
    if curve_type is None:
        curve_type == cfg.CURVAS_DE_PERSISTENCIA

    
    #Diccionario con las funciones a utilizar 
    curves = {
        "betti": betti_curve,
        "entropy": entropy_curve,
        "euler": euler_curve,
        "betti_combined": betti0_then_betti1
    }

    if curve_type not in curves:
        raise ValueError(
            f"curve_type debe ser uno de {list(curves.keys())}"
        )

    # Curvas que necesitan dimensión
    if curve_type in ("betti", "entropy"):
        if dimension is None:
            raise ValueError(
                f"La curva '{curve_type}' requiere que se especifique 'dimension'."
            )

        return curves[curve_type](
            intervals=intervals,
            dimension=dimension,
            times=times,
            **kwargs
        )

    # Curva de Euler (no usa dimension)
    else:
        return curves[curve_type](
            intervals=intervals,
            _=None,
            times=times
        )

    
# =============================================================================
# CURVAS DE PERSISTENCIA
# =============================================================================
#Partimos de que tenemos algo asi t | dimension | birth | death
#------------------------------------------------------------------------------    

# def betti_curve(intervals, dimension, times, **kwargs):
#     """
#     Calcula la curva de Betti para una dimensión dada.

#     Parameters
#     ----------
#     intervals : pandas.DataFrame
#         DataFrame que contiene los intervalos de persistencia.
#         Debe tener al menos las columnas:
#         - 'dimension' : dimensión homológica de cada intervalo
#         - 'birth'     : tiempo de nacimiento
#         - 'death'     : tiempo de muerte

#     dimension : int
#         Dimensión homológica para la cual se quiere calcular la curva de Betti
#         (por ejemplo, 0, 1, 2, ...).

#     times : array-like or None
#         Vector de tiempos en los que se evalúa la curva de Betti.
#         Si es None, se genera automáticamente un vector de 200 puntos
#         igualmente espaciados entre el menor 'birth' y el mayor 'death'
#         de la dimensión seleccionada.

#     **kwargs :
#         Argumentos adicionales (no se usan en esta implementación).
#         Se incluyen para compatibilidad con otras funciones o interfaces.

#     Returns
#     -------
#     times : numpy.ndarray
#         Vector de tiempos en los que se evaluó la curva.

#     betti : numpy.ndarray
#         Vector con el número de intervalos activos en cada tiempo.
#         Es decir, el número de clases topológicas vivas en cada instante.
#     """

#     # Se filtran únicamente los intervalos que pertenecen
#     # a la dimensión homológica solicitada
#     df = intervals[intervals["dimension"] == dimension]

#     # Si no se proporciona un vector de tiempos,
#     # se construye automáticamente uno
#     if times is None:
#         times = np.linspace(
#             df["birth"].min(),   # tiempo mínimo de nacimiento
#             df["death"].max(),   # tiempo máximo de muerte
#             200                  # número de puntos
#         )

#     # Lista donde se almacenará el valor de la curva de Betti
#     betti = []

#     # Para cada tiempo t se cuenta cuántos intervalos están "vivos"
#     for t in times:

#         # Un intervalo está activo en t si:
#         # birth <= t < death
#         count = ((df["birth"] <= t) & (t < df["death"])).sum()

#         # Se guarda el número de intervalos activos en ese tiempo
#         betti.append(count)

#     # Se devuelven los tiempos y la curva de Betti
#     return times, np.array(betti)


def betti_curve(intervals, dimension, times=None, **kwargs):
    """
    Calcula la curva de Betti para una dimensión dada, evitando NaN.
    """
    # Filtrar dimensión
    df = intervals[intervals["dimension"] == dimension]

    if df.empty:
        # Si no hay intervalos, devolver ceros
        if times is None:
            times = np.linspace(0, 1, 200)
        return times, np.zeros(len(times), dtype=int)

    # Crear vector de tiempos si no se proporciona
    if times is None:
        times = np.linspace(df["birth"].min(), df["death"].max(), 200)

    betti = []
    for t in times:
        count = ((df["birth"] <= t) & (t < df["death"])).sum()
        betti.append(count)

    return times, np.array(betti, dtype=int)


#------------------------------------------------------------------------------ 

# def entropy_curve(intervals, dimension, times, epsilon=1e-10):
#     """
#     Calcula la curva de entropía de persistencia a lo largo del tiempo
#     para una dimensión homológica dada.

#     Parameters
#     ----------
#     intervals : pandas.DataFrame
#         DataFrame que contiene los intervalos de persistencia.
#         Debe tener al menos las columnas:
#         - 'birth'     : tiempo de nacimiento
#         - 'death'     : tiempo de muerte
#         - 'dimension' : dimensión homológica del intervalo

#     dimension : int
#         Dimensión homológica para la cual se quiere calcular la entropía
#         (por ejemplo 0, 1, 2, ...).

#     times : array-like o None
#         Conjunto de instantes de tiempo en los que se evalúa la entropía.
#         Si es None, se genera automáticamente un conjunto de 200 tiempos
#         equiespaciados entre el menor birth y el mayor death.

#     epsilon : float, optional
#         Pequeño valor positivo que se suma dentro del logaritmo para evitar
#         problemas numéricos cuando p es muy pequeño.

#     Returns
#     -------
#     times : numpy.ndarray
#         Instantes de tiempo en los que se ha evaluado la entropía.

#     entropy : numpy.ndarray
#         Valores de la entropía en cada instante de tiempo.
#     """

#     # Filtramos los intervalos que pertenecen a la dimensión indicada
#     df = intervals[intervals["dimension"] == dimension]

#     # Si no se proporcionan tiempos, se crean 200 tiempos uniformes
#     # entre el menor nacimiento y la mayor muerte
#     if times is None:
#         times = np.linspace(df["birth"].min(), df["death"].max(), 200)

#     # Lista donde se almacenarán los valores de entropía
#     entropy = []

#     # Recorremos cada instante de tiempo
#     for t in times:

#         # Seleccionamos los intervalos "vivos" en el tiempo t
#         # Es decir, aquellos tales que birth <= t < death
#         alive = df[(df["birth"] <= t) & (t < df["death"])]

#         # Si no hay intervalos vivos, la entropía se define como 0
#         if len(alive) == 0:
#             entropy.append(0.0)
#             continue

#         # Longitudes de los intervalos vivos
#         lengths = alive["death"] - alive["birth"]

#         # Probabilidades normalizadas (suman 1)
#         p = lengths / lengths.sum()

#         # Entropía de Shannon asociada a estas probabilidades
#         entropy.append(-np.sum(p * np.log(p + epsilon)))

#     # Se devuelve el vector de tiempos y la curva de entropía
#     return times, np.array(entropy)


def entropy_curve(intervals, dimension, times=None, epsilon=1e-10):
    """
    Curva de entropía de persistencia, segura frente a NaN.
    """
    df = intervals[intervals["dimension"] == dimension]

    if df.empty:
        if times is None:
            times = np.linspace(0, 1, 200)
        return times, np.zeros(len(times))

    if times is None:
        times = np.linspace(df["birth"].min(), df["death"].max(), 200)

    entropy = []
    for t in times:
        alive = df[(df["birth"] <= t) & (t < df["death"])]
        if alive.empty:
            entropy.append(0.0)
            continue

        lengths = alive["death"] - alive["birth"]
        if lengths.sum() == 0:
            entropy.append(0.0)
            continue

        p = lengths / lengths.sum()
        entropy.append(-np.sum(p * np.log(p + epsilon)))

    return times, np.array(entropy)

#------------------------------------------------------------------------------ 

def euler_curve(intervals, _, times):
    """
    Calcula la curva característica de Euler a lo largo del tiempo
    a partir de un conjunto de intervalos de persistencia.

    Parameters
    ----------
    intervals : pandas.DataFrame
        DataFrame que contiene los intervalos de persistencia.
        Debe tener al menos las columnas:
        - 'birth'     : tiempo de nacimiento
        - 'death'     : tiempo de muerte
        - 'dimension' : dimensión homológica

    _ : cualquier valor (no usado)
        Este parámetro no se utiliza. Se mantiene únicamente para
        compatibilidad con una interfaz común con otras funciones
        (por ejemplo entropy_curve(intervals, dimension, times)).

    times : array-like o None
        Instantes de tiempo en los que se evalúa la característica
        de Euler.

        Si es None, se generan automáticamente 200 tiempos
        equiespaciados entre el menor birth y el mayor death.

    Returns
    -------
    times : numpy.ndarray
        Instantes de tiempo en los que se evalúa la curva.

    euler : numpy.ndarray
        Valores de la característica de Euler en cada instante.
    """

    # Si no se proporcionan tiempos, se construye un vector
    # de 200 instantes uniformemente espaciados
    if times is None:
        times = np.linspace(
            intervals["birth"].min(),
            intervals["death"].max(),
            200
        )

    # Lista donde se almacenará la curva de Euler
    euler = []

    # Para cada instante de tiempo t
    for t in times:

        # Valor de la característica de Euler en el tiempo t
        value = 0

        # Recorremos todas las dimensiones homológicas presentes
        # en el DataFrame
        for dim in intervals["dimension"].unique():

            # Seleccionamos únicamente los intervalos
            # de la dimensión dim
            df = intervals[intervals["dimension"] == dim]

            # Número de clases vivas en el tiempo t
            # (equivale al número de barras activas)
            beta = ((df["birth"] <= t) & (t < df["death"])).sum()

            # Contribución a la característica de Euler:
            # (-1)^dim * beta
            value += (-1) ** dim * beta

        # Guardamos el valor de la curva en el tiempo t
        euler.append(value)

    # Se devuelve el vector de tiempos y la curva de Euler
    return times, np.array(euler)   
    
#------------------------------------------------------------------------------     
# def betti0_then_betti1(intervals, times=None, **kwargs):
#     """
#     Genera una curva combinada donde primero están los valores
#     de Betti 0 y justo después los de Betti 1.

#     Parameters
#     ----------
#     intervals : pandas.DataFrame
#         DataFrame con los intervalos de persistencia.
#         Debe tener las columnas: 'dimension', 'birth', 'death'.

#     times : array-like or None
#         Vector de tiempos en los que se evaluarán las curvas.
#         Si es None, se genera automáticamente usando la dimensión 0.

#     **kwargs :
#         Argumentos adicionales que se pasan a betti_curve.

#     Returns
#     -------
#     times_combined : numpy.ndarray
#         Vector de tiempos repetido para cada dimensión.
#         (Primero todos los tiempos de β0, luego todos los tiempos de β1)

#     betti_combined : numpy.ndarray
#         Vector combinado con los valores de β0 y β1.
#         (Primero β0, luego β1)
#     """

#     # Curva de Betti 0
#     times0, betti0 = betti_curve(intervals, dimension=0, times=times, **kwargs)

#     # Curva de Betti 1 usando exactamente los mismos tiempos
#     _, betti1 = betti_curve(intervals, dimension=1, times=times0, **kwargs)

#     # Concatenamos
#     times_combined = np.concatenate([times0, times0])
#     betti_combined = np.concatenate([betti0, betti1])

#     return times_combined, betti_combined
    
def betti0_then_betti1(intervals, times=None, **kwargs):
    """
    Curva combinada β0 y β1 segura contra NaN.
    """
    times0, betti0 = betti_curve(intervals, dimension=0, times=times, **kwargs)
    _, betti1 = betti_curve(intervals, dimension=1, times=times0, **kwargs)

    times_combined = np.concatenate([times0, times0])
    betti_combined = np.concatenate([betti0, betti1])

    return times_combined, betti_combined
   
