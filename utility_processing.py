from typing import cast
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.figure import Figure
from pandas import DataFrame, Series
plt.style.use("default")

# ? Este archivo contiene funciones de utilidad utilizadas en los archivos de analisis '.ipynb'.
# ? Principalmente hay manipulación de Datagramas y ploteo de imagenes.

def quitar_asterisco(datagrama: DataFrame, mascara: list[str]) -> DataFrame:
    """ Separa un Datagrama según una máscara para la columna "Model".
    Luego además limpia asteríscos en estas columnas.

    :param DataFrame datagrama: Datagrama a limpiar.
    :param list[str] mascara: Lista de casos a incluir.
    :return DataFrame: Datagrama filtrado y limpiado.
    """
    nuevo_datagrama = datagrama[datagrama["Model"].isin(mascara)].copy()
    nuevo_datagrama["Model"] = nuevo_datagrama["Model"].str.replace("*", "", regex=False)
    return nuevo_datagrama


def crear_lista_diccionarios(datagram: list[DataFrame], datagrams: list[DataFrame], group: list[str], groups: list[str],
                             metrics: list[str], label: list[str], labels: list[str], order: list[str], xlabel: str,
                             ylabel: list[str], title: str | None = None, legend: bool = True, figsize=(7, 3)) -> list[dict[str, str | list[str] | DataFrame]]:
    """ Crea una lista de diccionarios para configurar gráficos basados en los parámetros proporcionados.

    :param list[DataFrame] datagram: Lista con un solo Datagrama.
    :param list[DataFrame] datagrams: Lista con dos Datagramas.
    :param list[str] group: Lista con un solo nombre de columna para agrupar.
    :param list[str] groups: Lista con dos nombres de columnas para agrupar.
    :param list[str] metrics: Lista de métricas a graficar.
    :param list[str] label: Lista con una sola etiqueta para el gráfico con un solo Datagrama.
    :param list[str] labels: Lista con dos etiquetas para el gráfico con dos Datagramas.
    :param list[str] order: Orden de las categorías para el grupo.
    :param str xlabel: Etiqueta para el eje X.
    :param list[str] ylabel: Lista de etiquetas para el eje Y, correspondiente a cada métrica.
    :param str title: Titulo de las figuras, por defecto es None.
    :param bool legend: Define si mostrar o no una leyenda, por defecto False.
    :param tuple figsize: Tamaño de la figura (ancho, alto) a usar al crear la figura.
    :return list[dict[str, str | list[str] | DataFrame]]: Lista de diccionarios con las configuraciones para poder plotear las figuras.
    """
    lista_diccionarios = []
    largo = len(datagrams)

    for i, metrica in enumerate(metrics):
        ylabel_actual = ylabel[i]

        # Diccionario para un solo datagrama
        dict_solo = {
            "datagrams": datagram,
            "metrics": [metrica],
            "groups_by": group,
            "order": order,
            "labels": label,
            "xlabel": xlabel,
            "ylabel": ylabel_actual,
            "title": title,
            "legend": legend,
            "figsize": figsize,
        }
        lista_diccionarios.append(dict_solo)

        # Diccionario para dos datagramas
        dict_multiple = {
            "datagrams": datagrams,
            "metrics": [metrica] * largo,
            "groups_by": groups,
            "order": order,
            "labels": labels,
            "xlabel": xlabel,
            "ylabel": ylabel_actual,
            "title": title,
            "legend": legend,
            "figsize": figsize,
        }
        lista_diccionarios.append(dict_multiple)

    return lista_diccionarios


def plot_metric(datagrams: DataFrame, metrics: list[str], groups_by: list[str], order: list[str], labels: list[str],
                xlabel: str | None = None, ylabel: str | None = None, title: str | None = None, legend: bool = False,
                figsize=(7, 3), fontsize: int = 10) -> Figure:
    """ Plotea y retorna una figura con curvas para un Datagrama entregado.
    Se realiza un scatter plot y además se dibuja una curva promedio para esos puntos.

    :param DataFrame datagrams: Datagrama con los datos a mostrar.
    :param list[str] metrics: Lista de métricas (columnas) a mostrar.
    :param list[str] groups_by: Lista de columnas para agrupar los datos. Es el eje X del plot.
    :param list[str] order: Lista de llaves con el que ordenar el eje X.
    :param list[str] labels: Nombre de cada curva. Solo se mostrará si legend es True.
    :param str xlabel: Nombre del eje X, por defecto None.
    :param str ylabel: Nombre del eje Y, por defecto None.
    :param str title: Titulo de la figura, por defecto None.
    :param bool legend: Define si mostrar o no una leyenda, por defecto False.
    :param int fontsize: Tamaño de fuente para etiquetas, ticks y leyenda.
    :return Figure: Figura matplotlib con las curvas dibujadas.
    """
    # Crear la figura como variable
    mi_figura = plt.figure(figsize=figsize)
    # Agregar un eje a la figura
    ax = mi_figura.add_subplot(111)

    # Definir una lista de colores para las curvas
    colors = sns.color_palette("hls", len(datagrams))

    # Iterar sobre los datagramas y plotear sus datos
    for i, (df, label, metric, group_by) in enumerate(zip(datagrams, labels, metrics, groups_by)):
        # Filtrar las filas que tengan NaN en la métrica seleccionada
        df = cast(DataFrame, df)
        df_filtered = df.dropna(subset=[metric])

        # Crear el scatter plot con barras de dispersión en el eje
        sns.stripplot(x=group_by, y=metric, data=df_filtered, order=order, jitter=True, 
                      marker='o', alpha=0.6, color=colors[i], ax=ax)

        # Calcular los promedios
        means = df_filtered.groupby(group_by, observed=True)[metric].mean().reindex(order)

        # Plotear la curva de los promedios en el eje
        ax.plot(order, means, marker='o', color=colors[i], label=label)

    # Configuración del gráfico usando el objeto ax
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize + 2)
    if legend:
        ax.legend(fontsize=fontsize)
    ax.tick_params(axis='x', rotation=45, labelsize=fontsize)  # Rotar etiquetas del eje x y tamaño
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.grid(True)
    mi_figura.tight_layout()  # Ajustar el diseño usando la figura

    plt.show()

    return mi_figura  # Devolver la figura para uso posterior


def comparar_metricas(datagrama: DataFrame, columna: str, casos_a_comparar: list[str] | tuple[str, str],
                      thr: float = 0.6) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """ Compara métricas numéricas entre dos casos específicos de una columna dada en un DataFrame.

    Esta función genera comparaciones entre dos casos (casos_a_comparar[0] y casos_a_comparar[1])
    para una columna específica, calculando diferencias absolutas, mejoras relativas, factores de mejora
    y reducción de error (para 'loss'). El orden en casos_a_comparar indica la dirección de la comparación:
    se evalúa la mejora desde casos_a_comparar[0] hacia casos_a_comparar[1].

    :param DataFrame datagrama: El DataFrame que contiene los datos a analizar.
    :param str columna: Nombre de la columna cuyos casos se desea comparar (e.g., "Optimizer").
    :param list[str] | tuple[str, str] casos_a_comparar: Lista o tupla con dos valores de la columna a comparar (e.g., ["AdamW", "SGD"]).
    :param float thr: Umbral para filtrar filas donde se tiene F1_score(M) <= thr., por defecto 0.6.
    :raises ValueError: Si casos_a_comparar no contiene exactamente dos elementos.
    :return tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]: Una tupla con seis DataFrames:
    
        - ``diff_df_all``: Diferencia absoluta (caso2 - caso1) para todas las filas válidas.
        - ``diff_df_thr``: Diferencia absoluta (caso2 - caso1) con umbral aplicado.
        - ``mejora_rel_df_all``: Mejora relativa en porcentaje para todas las filas válidas.
        - ``mejora_rel_df_thr``: Mejora relativa en porcentaje con umbral aplicado.
        - ``factor_df_all``: Factor de mejora (caso2 / caso1) para todas las filas válidas.
        - ``factor_df_thr``: Factor de mejora (caso2 / caso1) con umbral aplicado.
        - ``reduccion_error_df_all``: Reducción del error para "loss" (todas las filas válidas).
        - ``reduccion_error_df_thr``: Reducción del error para "loss" (con umbral aplicado).
    """
    # Definimos las métricas numéricas a comparar
    NUMERICAL_METRICS = [
        "precision(B)", "recall(B)", "F1_score(B)", "mAP50(B)", "mAP50-95(B)",
        "precision(M)", "recall(M)", "F1_score(M)", "mAP50(M)", "mAP50-95(M)",
        "fitness", "preprocess", "inference", "loss", "postprocess"
    ]

    NON_NUMERICAL_METRICS = ["Model", "Dataset", "Format", "Optimizer", "TransferLearning"]
    non_numerical_metrics_excluded = [item for item in NON_NUMERICAL_METRICS if item != columna]

    # Verificamos que casos_a_comparar tenga exactamente dos elementos
    if len(casos_a_comparar) != 2:
        raise ValueError("casos_a_comparar debe contener exactamente dos elementos.")

    caso1, caso2 = casos_a_comparar  # Desempaquetamos los casos a comparar

    # Creamos una tabla dinámica con las métricas numéricas, indexada por Model, Dataset y Format
    pivot = (datagrama.loc[:, NUMERICAL_METRICS + NON_NUMERICAL_METRICS]
             .pivot_table(index=non_numerical_metrics_excluded, columns=columna, values=NUMERICAL_METRICS))

    # Aplanamos los nombres de las columnas para que sean de la forma "métrica_caso"
    pivot.columns = [f"{metric}_{opt}" for metric, opt in pivot.columns]

    # Definimos máscaras para filtrar filas válidas
    # mask_all: filas donde ambos casos tienen F1_score(M) >= 0
    # mask_thr: filas donde ambos casos tienen F1_score(M) >= thr
    mask_all = (pivot[f"F1_score(M)_{caso1}"] >= 0) & (pivot[f"F1_score(M)_{caso2}"] >= 0)
    mask_thr = (pivot[f"F1_score(M)_{caso1}"] >= thr) & (pivot[f"F1_score(M)_{caso2}"] >= thr)

    # Aplicamos las máscaras al DataFrame pivoteado
    pivot_all = pivot[mask_all]
    pivot_thr = pivot[mask_thr]

    # Función auxiliar para calcular diferencias, mejoras relativas y factores de mejora
    def calcular_diferencias(df_pivot, metrics, caso1, caso2):
        diferencias = {metric: df_pivot[f"{metric}_{caso2}"] - df_pivot[f"{metric}_{caso1}"]
                       for metric in metrics}
        mejora_relativa = {metric: 100 * (df_pivot[f"{metric}_{caso2}"] - df_pivot[f"{metric}_{caso1}"]) / df_pivot[f"{metric}_{caso2}"]
                           for metric in metrics}
        factor_mejora = {metric: df_pivot[f"{metric}_{caso2}"] / df_pivot[f"{metric}_{caso1}"]
                         for metric in metrics}
        return diferencias, mejora_relativa, factor_mejora

    # Calculamos indicadores para ambas máscaras
    diff_all, rel_improv_all, factor_all = calcular_diferencias(pivot_all, NUMERICAL_METRICS, caso1, caso2)
    diff_thr, rel_improv_thr, factor_thr = calcular_diferencias(pivot_thr, NUMERICAL_METRICS, caso1, caso2)

    # Función auxiliar para calcular la reducción del error (solo para "loss")
    def calcular_reduccion_error(df_pivot, caso1, caso2, metric="loss"):
        return (df_pivot[f"{metric}_{caso1}"] - df_pivot[f"{metric}_{caso2}"]) / df_pivot[f"{metric}_{caso1}"]

    # Calculamos la reducción del error para "loss"
    reduccion_error_all = {"loss": calcular_reduccion_error(pivot_all, caso1, caso2, "loss")}
    reduccion_error_df_all = pd.DataFrame(reduccion_error_all, index=pivot_all.index).reset_index()

    reduccion_error_thr = {"loss": calcular_reduccion_error(pivot_thr, caso1, caso2, "loss")}
    reduccion_error_df_thr = pd.DataFrame(reduccion_error_thr, index=pivot_thr.index).reset_index()

    # Convertimos los resultados en DataFrames, preservando el índice como columnas
    diff_df_all = pd.DataFrame(diff_all, index=pivot_all.index).reset_index()
    diff_df_thr = pd.DataFrame(diff_thr, index=pivot_thr.index).reset_index()

    mejora_rel_df_all = pd.DataFrame(rel_improv_all, index=pivot_all.index).reset_index()
    mejora_rel_df_thr = pd.DataFrame(rel_improv_thr, index=pivot_thr.index).reset_index()

    factor_df_all = pd.DataFrame(factor_all, index=pivot_all.index).reset_index()
    factor_df_thr = pd.DataFrame(factor_thr, index=pivot_thr.index).reset_index()

    # Retornamos todos los DataFrames calculados
    return (diff_df_all, diff_df_thr,
            mejora_rel_df_all, mejora_rel_df_thr,
            factor_df_all, factor_df_thr,
            reduccion_error_df_all, reduccion_error_df_thr)


def crear_datagramas_filtrados(resultados: tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame],
                               columna: str, opciones: list[str], orden: list[str] | None = None
                               ) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """ Crea tablas de promedios, filtrados por valores de una columna y separados por casos con/sin outliers.

    :param tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame] resultados: Tupla con los Datagramas de comparaciones: 
    (``diff_all``, ``diff_thr``, ``mejora_all``, ``mejora_thr``, ``factor_all``, ``factor_thr``, ``red_err_all``, ``red_err_thr``)
    :param str columna: Nombre de la columna por la cual filtrar (e.g., "Format").
    :param list[str] opciones: Lista de valores únicos en la columna para filtrar (e.g., ["Pytorch", "TensorRT-F32", ...]).
    :param list[str] orden: Lista opcional, similar a ``opciones`` pero establece el orden de los datos, por defecto None.
    :return tuple[DataFrame, DataFrame, DataFrame, DataFrame]: Cuatro DataFrames tabulados:

        - ``tabla_diff``: Para diferencias absolutas.
        - ``tabla_diffr``: Para mejoras relativas.
        - ``tabla_diffm``: Para factores de mejora.
        - ``tabla_diffre``: Para reducción del error.
    """
    # Desempaquetar los resultados
    diff_all, diff_thr, mejora_all, mejora_thr, factor_all, factor_thr, red_err_all, red_err_thr = resultados

    # Función auxiliar para procesar un par de DataFrames (con y sin outliers)
    def procesar_par_df(df_all, df_thr, es_reduccion_error=False):
        promedios = []

        # 1) Filtrar y calcular promedios para cada opción
        for opcion in opciones:
            # Con outliers (df_all)
            filtro_all = df_all[df_all[columna] == opcion]
            mean_all = filtro_all.select_dtypes(include='number').mean()
            # 2) Añadir columnas de identificación
            mean_all[columna] = opcion
            mean_all["Outlier"] = True
            promedios.append(mean_all)

            # Sin outliers (df_thr)
            filtro_thr = df_thr[df_thr[columna] == opcion]
            mean_thr = filtro_thr.select_dtypes(include='number').mean()
            # 2) Añadir columnas de identificación
            mean_thr[columna] = opcion
            mean_thr["Outlier"] = False
            promedios.append(mean_thr)

        # Calcular promedios generales (Todos)
        mean_all_general = df_all.select_dtypes(include='number').mean()
        mean_all_general[columna] = "Todos"
        mean_all_general["Outlier"] = True
        promedios.append(mean_all_general)

        mean_thr_general = df_thr.select_dtypes(include='number').mean()
        mean_thr_general[columna] = "Todos"
        mean_thr_general["Outlier"] = False
        promedios.append(mean_thr_general)

        # 3) Tabular los datos
        tabla = pd.DataFrame(promedios)
        if orden is not None:
            categories = orden + ["Todos"]
        else:
            categories = opciones + ["Todos"]
        tabla[columna] = pd.Categorical(tabla[columna], categories=categories, ordered=True)
        tabla = tabla.sort_values(["Outlier", columna], ascending=[False, True])
        #if orden is not None:
        #    
        #else:
        #    tabla = pd.DataFrame(promedios).sort_values(by="Outlier", ascending=False)

        # Si es reducción del error, mantener solo la columna "loss"
        if es_reduccion_error:
            tabla = tabla[[columna, "Outlier", "loss"]]

        return tabla

    # Procesar cada tipo de comparación
    tabla_diff = procesar_par_df(diff_all, diff_thr)
    tabla_diffr = procesar_par_df(mejora_all, mejora_thr)
    tabla_diffm = procesar_par_df(factor_all, factor_thr)
    tabla_diffre = procesar_par_df(red_err_all, red_err_thr, es_reduccion_error=True)

    return tabla_diff, tabla_diffr, tabla_diffm, tabla_diffre


def crear_tabla_comparativa_para_formatos(diff_datagram_1: DataFrame, diff_datagram_2: DataFrame, dif_datagram_3: DataFrame) -> DataFrame:
    """ Crea una tabla comparativa entre tres Datagramas, excluyendo outliers. Usado para comparar Formatos de exportación.

    :param DataFrame diff_datagram_1: Datagrama de diferencias 1.
    :param DataFrame diff_datagram_2: Datagrama de diferencias 2.
    :param DataFrame dif_datagram_3: Datagrama de diferencias 3.
    :return DataFrame: Tabla de rendimientos ordenadas por formato.
    """
    columns_to_show = ["F1_score(M)", "mAP50(M)", "mAP50-95(M)", "fitness", "preprocess", "inference", "postprocess"]
    condition_1a = diff_datagram_1["Outlier"] == False
    condition_1b = diff_datagram_1["Model"] == "Todos"
    condition_2a = diff_datagram_2["Outlier"] == False
    condition_2b = diff_datagram_2["Model"] == "Todos"
    condition_3a = dif_datagram_3["Outlier"] == False
    condition_3b = dif_datagram_3["Model"] == "Todos"

    pytorch2f32 = diff_datagram_1.loc[(condition_1a) & (condition_1b), columns_to_show].reset_index(drop=True)
    pytorch2f32["Format"] = "TensorRT-F32"
    pytorch2f16 = diff_datagram_2.loc[(condition_2a) & (condition_2b), columns_to_show].reset_index(drop=True)
    pytorch2f16["Format"] = "TensorRT-F16"
    pytorch2int8 = dif_datagram_3.loc[(condition_3a) & (condition_3b), columns_to_show].reset_index(drop=True)
    pytorch2int8["Format"] = "TensorRT-INT8"

    return pd.concat([pytorch2f32, pytorch2f16, pytorch2int8])


def crear_tabla_comparativa_para_modelos_yolo(yolon: DataFrame, yolos: DataFrame, yolom: DataFrame, yolol: DataFrame, yolox: DataFrame,
                                              transfer_learning: bool = False, outlier: bool | str = False) -> DataFrame:
    """ Crea una tabla comparativa por modelos YOLO. Usada para comparar los resultados entre YOLOv8 y YOLO11.

    :param DataFrame yolon: Datagrama diferencias calculadas entre YOLOv8 y YOLO11 en tamaño n.
    :param DataFrame yolos: Datagrama diferencias calculadas entre YOLOv8 y YOLO11 en tamaño s.
    :param DataFrame yolom: Datagrama diferencias calculadas entre YOLOv8 y YOLO11 en tamaño m.
    :param DataFrame yolol: Datagrama diferencias calculadas entre YOLOv8 y YOLO11 en tamaño l.
    :param DataFrame yolox: Datagrama diferencias calculadas entre YOLOv8 y YOLO11 en tamaño x.
    :param bool transfer_learning: Determina el caso a incluir de la columna 'TransferLearning', por defecto False.
    :param bool | str outlier: Determina el caso a incluir de la columna 'Outlier', por defecto False.
    :return DataFrame: Datagrama con las comparaciones realizadas y ordenadas por modelo.
    """
    columns_to_show = ["F1_score(M)", "mAP50(M)", "mAP50-95(M)", "fitness", "preprocess", "inference", "postprocess"]

    # Diccionario de modelos
    modelos = {
        "YOLOn": yolon,
        "YOLOs": yolos,
        "YOLOm": yolom,
        "YOLOl": yolol,
        "YOLOx": yolox
    }

    # Lista para almacenar DataFrames procesados
    dfs = []
    for nombre, modelo in modelos.items():
        df = modelo.loc[(modelo["Outlier"] == outlier) & (modelo["TransferLearning"] == transfer_learning), columns_to_show].reset_index(drop=True)
        df["Model"] = nombre
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def apply_extra_conditions(df: DataFrame, conditions: dict) -> Series:
    """ Función auxiliar que evalúa una condición de pertenencia a una columna en un Datagrama.

    :param DataFrame df: Datagrama con los datos.
    :param dict conditions: Condicion de pertenencia a considerar.
    :return DataFrame: Datagrama de booleanos que mapean los casos a considerar.
    """
    condition = pd.Series(True, index=df.index)
    for key, value in conditions.items():
        condition = condition & (df[key].isin(value))
    return condition


def calcular_metricas(df: DataFrame, col_to_eval: str, casos: list[str], columns_to_show: list[str], extra_conditions: dict) -> dict[str, Series]:
    """ Función de utilidad que calcula la media, desviación estandar y el máximo de un Datagrama.

    :param DataFrame df: Datagrama con los datos a analizar.
    :param str col_to_eval: Columna a usar para evaluar las condiciones (e.g., "Model").
    :param list[str] casos: Casos a considerar (e.g., ["yolov9c-seg", "yolov9e-seg"]).
    :param list[str] columns_to_show: Columnas a mostrar en la Serie retornada (e.g., ["F1_score(M)", "mAP50(M)", "mAP50-95(M)"]).
    :param dict extra_conditions: Condiciones lógicas extras a considerar cuando se filtren los casos.
    :return dict[str, Series]: Serie con las medidas de media, desviación estandar y valor máximo.
    """
    metricas = {}
    for modelo in casos:
        condition = (df[col_to_eval] == modelo) & apply_extra_conditions(df, extra_conditions)
        mean_values = df.loc[condition, columns_to_show].mean()
        std_values = df.loc[condition, columns_to_show].std()
        max_values = df.loc[condition, columns_to_show].max()
        metricas[modelo] = {
            "mean": mean_values,
            "std": std_values,
            "max": max_values
        }
    return metricas


def metricas_a_dataframe(metricas: dict[str, Series], col_to_eval: str, casos: list[str]) -> DataFrame:
    """ Función de utilidad que convierte una Serie con métricas en un Datagrama y añade columnas.

    :param dict[str, Series] metricas: Diccionario de series con las métricas de media y dispersión.
    :param str col_to_eval: Columna a usar para evaluar las condiciones (e.g., "Dataset").
    :param list[str] casos: Casos a considerar (e.g., ["Deepfish", "Deepfish_LO"]).
    :return DataFrame: Datagrama con las métricas y columnas correspondientes.
    """
    data = []
    for modelo in casos:
        mean = metricas[modelo]["mean"]
        std = metricas[modelo]["std"]
        maximum = metricas[modelo]["max"]
        row = {
            f"{col_to_eval}": modelo,
            "F1_score(M)_max": maximum["F1_score(M)"],
            "F1_score(M)_mean": mean["F1_score(M)"],
            "F1_score(M)_std": std["F1_score(M)"],
            "mAP50(M)_max": maximum["mAP50(M)"],
            "mAP50(M)_mean": mean["mAP50(M)"],
            "mAP50(M)_std": std["mAP50(M)"],
            "mAP50-95(M)_max": maximum["mAP50-95(M)"],
            "mAP50-95(M)_mean": mean["mAP50-95(M)"],
            "mAP50-95(M)_std": std["mAP50-95(M)"]
        }
        data.append(row)
    return pd.DataFrame(data)


def crear_tabla_promedios_modelos_yolo(df: DataFrame, extra_conditions: dict[str, list] | None = None) -> tuple[DataFrame, DataFrame, DataFrame]:
    """ Cálcula métricas de media y dispersión para los diferentes resultados de modelos YOLO dentro de un datagrama.
    Se usa para comparar los modelos entre sí.

    :param DataFrame df: Datagrama con las métricas de validación de los modelos.
    :param dict[str, list] extra_conditions: Diccionario con condiciones extras para filtrar el datagrama (e.g., {"Format": ["Pytorch"], "TransferLearning": [True]}), por defecto None.
    :return tuple[DataFrame, DataFrame, DataFrame]: Tres datagramas tabulados:

        - ``df_yolov8_metrics``: Metricas para YOLOv8.
        - ``df_yolov9_metrics``: Metricas para YOLOv9.
        - ``df_yolo11_metrics``: Metricas para YOLO11.
    """
    # Si no se proporciona extra_conditions, usar un diccionario vacío
    if extra_conditions is None:
        extra_conditions = {}

    # Columnas a analizar
    columns_to_show = ["F1_score(M)", "mAP50(M)", "mAP50-95(M)"]

    # Listas de modelos
    modelos_yolov8 = ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"]
    modelos_yolov9 = ["yolov9c-seg", "yolov9e-seg"]
    modelos_yolo11 = ["yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"]

    # Calcular métricas para YOLOv8 y YOLO11
    metricas_yolov8 = calcular_metricas(df, "Model", modelos_yolov8, columns_to_show, extra_conditions)
    metricas_yolov9 = calcular_metricas(df, "Model", modelos_yolov9, columns_to_show, extra_conditions)
    metricas_yolo11 = calcular_metricas(df, "Model", modelos_yolo11, columns_to_show, extra_conditions)

    # Crear DataFrames finales
    df_yolov8_metrics = metricas_a_dataframe(metricas_yolov8, "Model", modelos_yolov8)
    df_yolov9_metrics = metricas_a_dataframe(metricas_yolov9, "Model", modelos_yolov9)
    df_yolo11_metrics = metricas_a_dataframe(metricas_yolo11, "Model", modelos_yolo11)

    return df_yolov8_metrics, df_yolov9_metrics, df_yolo11_metrics


def crear_tabla_promedios_dataset(df: DataFrame, datasets: list[str], extra_conditions: dict[str, list] | None = None) -> DataFrame:
    """ Cálcula métricas de media y dispersión para los diferentes datasets dentro de un datagrama.
    Se usa para comparar los datasets con y sin imagenes de fondo.

    :param DataFrame df: Datagrama con las métricas de validación de los modelos.
    :param dict[str, list] extra_conditions: Diccionario con condiciones extras para filtrar el Datagrama (e.g., {"Format": ["Pytorch"], "TransferLearning": [True]}), por defecto None.
    :return DataFrame: Datagrama tabulado con comparaciones por dataset.
    """
    # Si no se proporciona extra_conditions, usar un diccionario vacío
    if extra_conditions is None:
        extra_conditions = {}

    # Columnas a analizar
    columns_to_show = ["F1_score(M)", "mAP50(M)", "mAP50-95(M)"]

    # Calcular métricas para YOLOv8 y YOLO11
    metricas_dataset = calcular_metricas(df, "Dataset", datasets, columns_to_show, extra_conditions)

    # Crear DataFrames finales
    df_dataset = metricas_a_dataframe(metricas_dataset, "Dataset", datasets)

    return df_dataset


def crear_tablas_mejora_tiempos(
    df: DataFrame,
    formatos: list[str] | None = None,
    remove_outliers: bool = False,
    outlier_groupby: list[str] | None = None,
    outlier_method: str = "iqr",
    iqr_k: float = 1.5,
    z_thresh: float = 3.0,
    baseline_only: bool = True,
    min_group_size: int = 3,
) -> dict[str, DataFrame]:
    """Crea una tabla por formato que compara tiempos/FPS entre Pytorch y TensorRT.
    Empareja por (Dataset, Optimizer, Model) y calcula  factor de mejora para cada Model:
      - Inference (Factor de Mejora): promedio de PT / TRT [>1 es mejor]
      - Inference (Máx diff): máximo inference_factor por modelo
      - FPS (Factor de Mejora): promedio de TRT / PT [>1 es mejor]
      - FPS (Máx diff): rango de factores = max(TRT/PT) - min(TRT/PT)

    Soporta exclusión de outliers en 'inference' antes del emparejamiento.

    :param df: Debe incluir columnas: 'Format','Dataset','Optimizer','Model','inference','FPS'
    :param formatos: ["TensorRT-F32","TensorRT-F16","TensorRT-INT8"] por defecto.
    :param remove_outliers: Si True, filtra outliers en 'inference' por grupos.
    :param outlier_groupby: Claves de agrupación para detectar outliers. Por defecto ["Model","Format"].
    :param outlier_method: "iqr" o "zscore".
    :param iqr_k: Multiplicador IQR (1.5 por defecto).
    :param z_thresh: Umbral |z| para z-score (3.0 por defecto).
    :param baseline_only: Si True, filtra outliers solo en Pytorch. Si False, filtra en Pytorch y formatos TRT.
    :param min_group_size: Tamaño mínimo de grupo para activar el filtro (si no, no se filtra).
    """

    if formatos is None:
        formatos = ["TensorRT-F32", "TensorRT-F16", "TensorRT-INT8"]

    required_cols = {"Format", "Dataset", "Optimizer", "Model", "inference", "FPS"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el DataFrame: {missing}")

    # Utilidad: filtrar outliers en 'inference' por grupos
    def _filtrar_outliers_por_grupo(dfin: DataFrame, group_keys: list[str]) -> DataFrame:
        if dfin.empty:
            return dfin
        sname = "inference"

        def mask_group(g: DataFrame) -> pd.Series:
            s = g[sname]
            if len(s) < min_group_size:
                return pd.Series(True, index=g.index)
            if outlier_method == "iqr":
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                if pd.isna(iqr) or iqr == 0:
                    return pd.Series(True, index=g.index)
                low = q1 - iqr_k * iqr
                high = q3 + iqr_k * iqr
                return (s >= low) & (s <= high)
            elif outlier_method == "zscore":
                mu = s.mean()
                sigma = s.std(ddof=0)
                if pd.isna(sigma) or sigma == 0:
                    return pd.Series(True, index=g.index)
                z = (s - mu) / sigma
                return z.abs() <= z_thresh
            else:
                return pd.Series(True, index=g.index)

        gb = dfin.groupby(group_keys, group_keys=False)
        # Detectar en tiempo de ejecución si 'apply' acepta el argumento 'include_groups'
        import inspect
        kwargs = {}
        try:
            sig = inspect.signature(gb.apply)
            if 'include_groups' in sig.parameters:
                kwargs['include_groups'] = False
        except (ValueError, TypeError):
            # Si no se puede obtener la firma (método C o similar), fallback sin kwargs
            kwargs = {}
        # Llamar a apply con kwargs dinámicos evita errores de tipo estático por argumentos desconocidos
        mask_raw = gb.apply(mask_group, **kwargs)

        # Normalizar a Series booleana alineada
        if isinstance(mask_raw, pd.DataFrame):
            mask = mask_raw.iloc[:, 0]
        else:
            mask = mask_raw

        if isinstance(mask.index, pd.MultiIndex):
            mask.index = mask.index.get_level_values(-1)

        mask = mask.astype(bool).reindex(dfin.index, fill_value=True)
        return dfin.loc[mask]

    # Copia y filtrado opcional de outliers
    df_work = df.copy()
    if remove_outliers:
        if outlier_groupby is None:
            outlier_groupby = ["Model", "Format"]

        if baseline_only:
            m_pt = df_work["Format"] == "Pytorch"
            df_work = pd.concat(
                [
                    _filtrar_outliers_por_grupo(df_work.loc[m_pt], outlier_groupby),
                    df_work.loc[~m_pt],
                ],
                ignore_index=True,
            )
        else:
            m_comp = df_work["Format"].isin(["Pytorch"] + formatos)
            df_work = pd.concat(
                [
                    _filtrar_outliers_por_grupo(df_work.loc[m_comp], outlier_groupby),
                    df_work.loc[~m_comp],
                ],
                ignore_index=True,
            )

    # Base Pytorch
    df_pt = (
        df_work[df_work["Format"] == "Pytorch"]
        .loc[:, ["Dataset", "Optimizer", "Model", "inference", "FPS"]]
        .rename(columns={"inference": "inference_pt", "FPS": "fps_pt"})
    )

    tablas: dict[str, DataFrame] = {}

    columns_to_rename = {
        "Model": "Models",
        "inference_factor": "Inference",
        "inference_factor_max_diff": "Inference (Máx diff)",
        "fps_diff": "FPS",
        "fps_max_diff": "FPS (Máx diff)"
    }

    final_columns = ["Models", "Inference", "Inference (Máx diff)",
        "FPS", "FPS (Máx diff)"
    ]

    for fmt in formatos:
        df_fmt = (
            df_work[df_work["Format"] == fmt]
            .loc[:, ["Dataset", "Optimizer", "Model", "inference", "FPS"]]
            .rename(columns={"inference": "inference_fmt", "FPS": "fps_fmt"})
        )

        # Emparejar Pytorch con el formato por (Dataset, Optimizer, Model)
        pairs = df_pt.merge(df_fmt, on=["Dataset", "Optimizer", "Model"], how="inner")
        if pairs.empty:
            tablas[fmt] = pd.DataFrame(columns=final_columns)
            continue

        # Evitar divisiones por cero
        den_inf_pt = pairs["inference_pt"].replace(0, pd.NA)
        den_inf_fmt = pairs["inference_fmt"].replace(0, pd.NA)
        den_fps_pt = pairs["fps_pt"].replace(0, pd.NA)
        den_fps_fmt = pairs["fps_fmt"].replace(0, pd.NA)

        # Métricas por par
        pairs["inference_factor"] = den_inf_fmt / den_inf_pt
        pairs["fps_diff"] = den_fps_fmt - den_fps_pt

        # Agregaciones por Model
        mean_agg = (
            pairs.groupby("Model", as_index=False)
            .agg(inference_factor=("inference_factor", "mean"),
                fps_diff=("fps_diff", "mean"))
        )

        range_agg = (
            pairs.groupby("Model", as_index=False)
            .agg(inference_factor_max=("inference_factor", "min"),
                fps_max_diff=("fps_diff", "max"))
        )

        agg = mean_agg.merge(range_agg, on="Model", how="left")
        # Máx diff (inference): usar el máximo factor directamente
        agg["inference_factor_max_diff"] = agg["inference_factor_max"]
        # FPS (Máx diff) se mantiene como rango max - min
        agg["fps_max_diff"] = agg["fps_max_diff"]

        result = (
            agg.rename(columns=columns_to_rename)
            .loc[:, final_columns]
            .sort_values("Models")
            .reset_index(drop=True)
        )

        tablas[fmt] = result

    return tablas


def guardar_figura(fig: Figure, nombre: str, extension: str | None = "png", directorio: str | None = None,
                   dpi: int = 300, bbox_inches: str = "tight") -> str:
    """Guarda una figura matplotlib en disco.

    :param Figure fig: Objeto figura a guardar.
    :param str nombre: Nombre base del archivo (puede incluir o no extensión).
    :param str | None extension: Extensión deseada (p.ej. 'png', 'pdf'). Si None se usa la extensión incluida en nombre
                                 o 'png' por defecto.
    :param str | None directorio: Directorio donde guardar. Si None se guarda en el directorio actual.
    :param int dpi: Resolución en DPI para formatos raster.
    :param str bbox_inches: Parámetro bbox_inches pasado a savefig.
    :return str: Ruta absoluta del archivo guardado.
    """
    from pathlib import Path

    # Preparar directorio
    dir_path = Path(directorio) if directorio is not None else Path.cwd()
    dir_path.mkdir(parents=True, exist_ok=True)

    # Determinar extensión
    base = Path(nombre)
    if extension is None:
        if base.suffix:
            ext = base.suffix.lstrip(".")
            filename = base.name
        else:
            ext = "png"
            filename = base.name
    else:
        ext = extension.lstrip(".")
        filename = base.stem

    target = dir_path / f"{filename}.{ext}"

    # Guardar la figura
    fig.savefig(target, dpi=dpi, bbox_inches=bbox_inches)
    return str(target.resolve())