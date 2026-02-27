# =============================================================================
# LIBRERIAS
# =============================================================================
import os
import pandas as pd
import numpy as np
from pathlib import Path

##Script mios
import cfg as cfg
from organizacion_data import *
from curvas_persistencia import *
from crocker_plot_betti import *

# =============================================================================
# PROCESADO
# =============================================================================
def procesar_evacuaciones_genial(
    data_archivos,
    ids, 
    etiqueta,  # empujando, sinempujar
    ruta_salida,
    n_t=None,
    salto=None,
    mymaxscale=None,
    numepsilon=None,
    curve_type=None,   # "betti", "entropy", "euler", "betti0_then_betti1"
    dim=None,                # Solo se usa si curve_type es "betti" o "entropy"
    **curve_kwargs
):
    """
    Guarda curvas topológicas individuales por evacuación/ventana.

    Parámetros
    ----------
    data_archivos : pd.DataFrame
        Datos de la evacuación completa.
    ids : array-like
        Lista de IDs de evacuación.
    etiqueta : str
        Etiqueta del grupo (ej. "empujando").
    ruta_salida : str / Path
        Carpeta donde se guardarán los CSV.
    n_t : int, opcional
        Tamaño de ventana (frames consecutivos). Por defecto, cfg.NUM_FRAMES.
    salto : int, opcional
        Salto entre frames. Por defecto, cfg.SALTO.
    mymaxscale : float, opcional
        Escala máxima para construir el complejo.
    numepsilon : int, opcional
        Número de epsilons para la grilla. Por defecto, cfg.NUMEPSILON.
    curve_type : str
        Tipo de curva: "betti", "entropy", "euler", "betti0_then_betti1".
    dim : int
        Dimensión de la curva si aplica.
    **curve_kwargs : dict
        Parámetros adicionales para las curvas.
    """


    # -----------------------------
    # Parámetros por defecto
    # -----------------------------
    if n_t is None:
        n_t = cfg.NUM_FRAMES
    if salto is None:
        salto = cfg.SALTO
    if mymaxscale is None:
        mymaxscale = cfg.MYMAXSCALE
    if numepsilon is None:
        numepsilon = cfg.NUMEPSILON
    if curve_type is None:
        curve_type = cfg.CURVA_DE_PERSISTENCIA
    if dim is None:
        dim = cfg.DIMENSION
        

    os.makedirs(ruta_salida, exist_ok=True)

    # -----------------------------
    # Loop por experimento (evacuación)
    # -----------------------------
    for ev in ids:

        data_ev = data_archivos[data_archivos[cfg.ID_COL] == ev]

        # Ventanas de frames, lista de dataframes
        ventanas = [data_ev] if n_t == 1 else datos_consecutivos(data_ev, n_t)
        
        for i, data in enumerate(ventanas, start=1):

            frames = np.sort(data[cfg.FRAME_COL].unique())
            interval_time = frames[::salto]

            homologydata = []

            for time in interval_time:

                mydata = data.loc[
                    data[cfg.FRAME_COL] == time,
                    [cfg.X_COL, cfg.Y_COL]
                ]

                newdata = getIntervals(mydata, thistime=time)
                df_time = pd.DataFrame(newdata)

                if not df_time.empty:
                    homologydata.append(df_time)

            if not homologydata:
                continue

            homologydata = pd.concat(homologydata, ignore_index=True)
            
            
            # -----------------------------
            # Generar la curva según el tipo
            # -----------------------------
            griddata = turnIntervalsIntoGrid(
                            homologydata,
                            curve_type
                        )

            # -----------------------------
            # Generar la matriz con las curvas
            # -----------------------------
            matriz_curvas = crear_matrices(griddata, dim)
            
            # -----------------------------
            # Guardar
            # -----------------------------
            
            # Nombre del archivo
            nombre_archivo = f"{curve_type}_{dim}_{ev}_{n_t}_{i}.csv"
            
            # Carpeta de salida
            folder_name = ruta_salida / f"{curve_type}_{dim}"
            folder_name.mkdir(parents=True, exist_ok=True)  # equivalente a os.makedirs
            
            # Ruta completa del archivo
            file_path = folder_name / nombre_archivo
            
            # Guardar CSV
            matriz_curvas.to_csv(file_path, index=False)
            
            print(f"Guardado: {nombre_archivo}")

#No esta con la eleccion de curvas 

# def procesar_evacuaciones(data_archivos,
#                            ids,
#                            etiqueta,      # "empujando" o "sin_empujar"
#                            rutas_base,
#                            n_t = None,    # None -> un frame | 100 -> 4 segundos
#                            salto = None,
#                            L=None,
#                            mymaxdimension= None,
#                            mymaxscale = None,
#                            numepsilon = None):
 
#     # ---------------------------------------------------------
#     # Parametros por defecto
#     # ---------------------------------------------------------   
#     if n_t is None:
#         n_t =cfg.NUM_FRAMES
#     if salto is None:
#         salto = cfg.SALTO
#     if L is None:
#         L =cfg.L
#     if mymaxdimension is None:
#         mymaxdimension = cfg.MYMAXDIMENSION
#     if mymaxscale is None:
#         mymaxscale = cfg.MYMAXSCALE
#     if numepsilon is None:
#         numepsilon =cfg.NUMEPSILON


#     for ev in ids:
#         print(ev)

#         data_ev = data_archivos[
#             data_archivos[cfg.ID_COL] == ev
#         ]

#         # ---------------------------------------------------------
#         # Selección de ventanas
#         # ---------------------------------------------------------
#         if(n_t == 1):
#             # Caso: toda la evacuación
#             lista_df = [data_ev]
#         else:
#             # Caso: ventanas de n_t frames
#             lista_df = datos_consecutivos(data_ev, n_t)

#         # ---------------------------------------------------------
#         # Procesar cada ventana y calculamos la homologia 
#         # ---------------------------------------------------------
#         for i, data in enumerate(lista_df, start=1):

#             print(i)
            
#             #Creamos una variables que se queda con los frames que hay en cada df
#             frames = sorted(data[cfg.FRAME_COL].unique())
#             #Esta varible nos da los saltos
#             interval_time = frames[::salto]

#             homologydata = []

#             for time in interval_time:

#                 mydata = data.loc[
#                     data[cfg.FRAME_COL] == time,
#                     [cfg.X_COL, cfg.Y_COL]
#                 ]

#                 newdata = getIntervals(
#                     mydata,
#                     thistime=time
#                 )
                
#                 # newdata es una lista de diccionarios
#                 df_time = pd.DataFrame(newdata)

#                 homologydata.append(df_time)

#             homologydata = pd.concat(homologydata, ignore_index=True)
            

#             # -----------------------------------------------------
#             # Betti grid
#             # -----------------------------------------------------
#             griddata = turnIntervalsIntoGrid(
#                 homologydata
#             )

#             # -----------------------------------------------------
#             # Betti 0
#             # -----------------------------------------------------
#             betti_0 = (
#                 griddata[griddata["dimension"] == 0]
#                 .drop(columns=["dimension"])
#                 .pivot(index="epsilon", columns="t", values="betticount")
#             )

#             # -----------------------------------------------------
#             # Betti 1
#             # -----------------------------------------------------
#             betti_1 = (
#                 griddata[griddata["dimension"] == 1]
#                 .drop(columns=["dimension"])
#                 .pivot(index="epsilon", columns="t", values="betticount")
#             )

#             # -----------------------------------------------------
#             # Matriz conjunta y Euler
#             # -----------------------------------------------------
#             betti = pd.concat([betti_0, betti_1], axis=0)
#             betti_euler = betti_0 - betti_1

#             # -----------------------------------------------------
#             # Rutas
#             # -----------------------------------------------------
#             ruta_m0 = rutas_base["b0"]
#             ruta_m1 = rutas_base["b1"]
#             ruta_me = rutas_base["be"]
#             ruta_m  = rutas_base["b"]

#             for r in [ruta_m0, ruta_m1, ruta_me, ruta_m]:
#                 os.makedirs(r, exist_ok=True)

#             # -----------------------------------------------------
#             # Nombre de archivo
#             # -----------------------------------------------------
#             if n_t is None:
#                 sufijo = f"{ev}"
#             else:
#                 sufijo = f"{ev}_{i}"

#             betti_0.to_csv(os.path.join(ruta_m0, f"mb0_{sufijo}.csv"),
#                             index=False)
#             betti_1.to_csv(os.path.join(ruta_m1, f"mb1_{sufijo}.csv"),
#                             index=False)
#             betti_euler.to_csv(os.path.join(ruta_me, f"mbe_{sufijo}.csv"),
#                                 index=False)
#             betti.to_csv(os.path.join(ruta_m, f"mb_{sufijo}.csv"),
#                           index=False)
