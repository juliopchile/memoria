import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def quitar_asterisco(datagrama, mascara):
    nuevo_datagrama = datagrama[datagrama["Model"].isin(mascara)].copy()
    nuevo_datagrama["Model"] = nuevo_datagrama["Model"].str.replace("*", "", regex=False)
    return nuevo_datagrama


def crear_lista_diccionarios(datagram, datagrams, group, groups, metrics, label, labels, order, xlabel, ylabel, title=None, legend=True):
    """
    Crea una lista de diccionarios para configurar gráficos basados en los parámetros proporcionados.

    Parámetros:
    - datagram: Lista con un solo DataFrame.
    - datagrams: Lista con dos DataFrames.
    - group: Lista con un solo nombre de columna para agrupar.
    - groups: Lista con dos nombres de columnas para agrupar.
    - metrics: Lista de métricas a graficar.
    - label: Lista con una sola etiqueta para el gráfico con un solo datagrama.
    - labels: Lista con dos etiquetas para el gráfico con dos datagramas.
    - order: Orden de las categorías para el grupo.
    - xlabel: Etiqueta para el eje X.
    - ylabel: Lista de etiquetas para el eje Y, correspondiente a cada métrica.

    Retorna:
    - Una lista de diccionarios, cada uno configurado para un gráfico específico.
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
        }
        lista_diccionarios.append(dict_multiple)

    return lista_diccionarios


def plot_metric(datagrams, metrics, groups_by, order, labels, xlabel=None, ylabel=None, title=None, legend=False):
    # Crear la figura como variable
    mi_figura = plt.figure(figsize=(8, 4))
    # Agregar un eje a la figura
    ax = mi_figura.add_subplot(111)

    # Definir una lista de colores para las curvas
    colors = sns.color_palette("hls", len(datagrams))

    # Iterar sobre los datagramas y plotear sus datos
    for i, (df, label, metric, group_by) in enumerate(zip(datagrams, labels, metrics, groups_by)):
        # Filtrar las filas que tengan NaN en la métrica seleccionada
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
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend()
    ax.tick_params(axis='x', rotation=45)  # Rotar etiquetas del eje x
    ax.grid(True)
    mi_figura.tight_layout()  # Ajustar el diseño usando la figura

    plt.show()

    return mi_figura  # Devolver la figura para uso posterior


def comparar_metricas(datagrama, columna, casos_a_comparar, thr=0.6):
    """
    Compara métricas numéricas entre dos casos específicos de una columna dada en un DataFrame.

    Esta función genera comparaciones entre dos casos (casos_a_comparar[0] y casos_a_comparar[1])
    para una columna específica, calculando diferencias absolutas, mejoras relativas, factores de mejora
    y reducción de error (para 'loss'). El orden en casos_a_comparar indica la dirección de la comparación:
    se evalúa la mejora desde casos_a_comparar[0] hacia casos_a_comparar[1].

    Parámetros:
    -----------
    datagrama : pd.DataFrame
        El DataFrame que contiene los datos a analizar.
    columna : str
        Nombre de la columna cuyos casos se desea comparar (e.g., "Optimizer").
    casos_a_comparar : list o tuple
        Lista o tupla con dos valores de la columna a comparar (e.g., ["AdamW", "SGD"]).
        El orden importa: se compara de casos_a_comparar[0] a casos_a_comparar[1].
    thr : float, opcional (default=0.6)
        Umbral para filtrar filas donde ambos casos tienen F1_score(M) >= thr.

    Retorna:
    --------
    tuple
        Una tupla con seis DataFrames:
        - diff_df_all: Diferencia absoluta (caso2 - caso1) para todas las filas válidas.
        - diff_df_thr: Diferencia absoluta (caso2 - caso1) con umbral aplicado.
        - mejora_rel_df_all: Mejora relativa en porcentaje para todas las filas válidas.
        - mejora_rel_df_thr: Mejora relativa en porcentaje con umbral aplicado.
        - factor_df_all: Factor de mejora (caso2 / caso1) para todas las filas válidas.
        - factor_df_thr: Factor de mejora (caso2 / caso1) con umbral aplicado.
        - reduccion_error_df_all: Reducción del error para "loss" (todas las filas válidas).
        - reduccion_error_df_thr: Reducción del error para "loss" (con umbral aplicado).

    Raises:
    -------
    ValueError
        Si casos_a_comparar no contiene exactamente dos elementos.
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
    pivot = (
        datagrama.loc[:, NUMERICAL_METRICS + NON_NUMERICAL_METRICS]
        .pivot_table(index=non_numerical_metrics_excluded, columns=columna, values=NUMERICAL_METRICS)
    )

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
        diferencias = {
            metric: df_pivot[f"{metric}_{caso2}"] - df_pivot[f"{metric}_{caso1}"]
            for metric in metrics
        }
        mejora_relativa = {
            metric: 100 * (df_pivot[f"{metric}_{caso2}"] - df_pivot[f"{metric}_{caso1}"]) / df_pivot[f"{metric}_{caso2}"]
            for metric in metrics
        }
        factor_mejora = {
            metric: df_pivot[f"{metric}_{caso2}"] / df_pivot[f"{metric}_{caso1}"]
            for metric in metrics
        }
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
    return (
        diff_df_all, diff_df_thr,
        mejora_rel_df_all, mejora_rel_df_thr,
        factor_df_all, factor_df_thr,
        reduccion_error_df_all, reduccion_error_df_thr
    )


def crear_datagramas_filtrados(resultados, columna, opciones, orden=None):
    """
    Crea tablas de promedios filtrados por valores de una columna y casos con/sin outliers.

    Parámetros:
    -----------
    resultados : tuple
        Tupla con los DataFrames de comparaciones:
        (diff_all, diff_thr, mejora_all, mejora_thr, factor_all, factor_thr, red_err_all, red_err_thr)
    columna : str
        Nombre de la columna por la cual filtrar (e.g., "Format").
    opciones : list
        Lista de valores únicos en la columna para filtrar (e.g., ["Pytorch", "TensorRT-F32", ...]).

    Retorna:
    --------
    tuple
        Cuatro DataFrames tabulados:
        - tabla_diff: Para diferencias absolutas.
        - tabla_diffr: Para mejoras relativas.
        - tabla_diffm: Para factores de mejora.
        - tabla_diffre: Para reducción del error.
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


def crear_tabla_comparativa_para_formatos(diff_datagram_1, diff_datagram_2, dif_datagram_3):
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


def crear_tabla_comparativa_para_modelos_yolo(yolon, yolos, yolom, yolol, yolox, transfer_learning=False, outlier=False):
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


def crear_tabla_promedios_modelos_yolo(df, extra_conditions=None):
    # Si no se proporciona extra_conditions, usar un diccionario vacío
    if extra_conditions is None:
        extra_conditions = {}

    # Columnas a analizar
    columns_to_show = ["F1_score(M)", "mAP50(M)", "mAP50-95(M)"]

    # Listas de modelos
    modelos_yolov8 = ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"]
    modelos_yolov9 = ["yolov9c-seg", "yolov9e-seg"]
    modelos_yolo11 = ["yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"]

    # Función auxiliar para aplicar condiciones extras
    def apply_extra_conditions(df, conditions):
        condition = True  # Base inicial
        for key, value in conditions.items():
            condition = condition & (df[key].isin(value))  # Combinar condiciones
        return condition

    # Función para calcular medias y desviaciones estándar
    def calcular_metricas(df, modelos, extra_conditions):
        metricas = {}
        for modelo in modelos:
            condition = (df["Model"] == modelo) & apply_extra_conditions(df, extra_conditions)
            mean_values = df.loc[condition, columns_to_show].mean()
            std_values = df.loc[condition, columns_to_show].std()
            max_values = df.loc[condition, columns_to_show].max()
            metricas[modelo] = {
                "mean": mean_values,
                "std": std_values,
                "max": max_values
            }
        return metricas

    # Calcular métricas para YOLOv8 y YOLO11
    metricas_yolov8 = calcular_metricas(df, modelos_yolov8, extra_conditions)
    metricas_yolov9 = calcular_metricas(df, modelos_yolov9, extra_conditions)
    metricas_yolo11 = calcular_metricas(df, modelos_yolo11, extra_conditions)

    # Función para convertir métricas en DataFrame
    def metricas_a_dataframe(metricas, modelos):
        data = []
        for modelo in modelos:
            mean = metricas[modelo]["mean"]
            std = metricas[modelo]["std"]
            maximum = metricas[modelo]["max"]
            row = {
                "Model": modelo,
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

    # Crear DataFrames finales
    df_yolov8_metrics = metricas_a_dataframe(metricas_yolov8, modelos_yolov8)
    df_yolov9_metrics = metricas_a_dataframe(metricas_yolov9, modelos_yolov9)
    df_yolo11_metrics = metricas_a_dataframe(metricas_yolo11, modelos_yolo11)

    return df_yolov8_metrics, df_yolov9_metrics, df_yolo11_metrics


def crear_tabla_promedios_dataset(df, datasets, extra_conditions=None):
    # Si no se proporciona extra_conditions, usar un diccionario vacío
    if extra_conditions is None:
        extra_conditions = {}

    # Columnas a analizar
    columns_to_show = ["F1_score(M)", "mAP50(M)", "mAP50-95(M)"]

    # Función auxiliar para aplicar condiciones extras
    def apply_extra_conditions(df, conditions):
        condition = True  # Base inicial
        for key, value in conditions.items():
            condition = condition & (df[key].isin(value))  # Combinar condiciones
        return condition

    # Función para calcular medias y desviaciones estándar
    def calcular_metricas(df, datasets, extra_conditions):
        metricas = {}
        for dataset in datasets:
            condition = (df["Dataset"] == dataset) & apply_extra_conditions(df, extra_conditions)
            mean_values = df.loc[condition, columns_to_show].mean()
            std_values = df.loc[condition, columns_to_show].std()
            max_values = df.loc[condition, columns_to_show].max()
            metricas[dataset] = {
                "mean": mean_values,
                "std": std_values,
                "max": max_values
            }
        return metricas

    # Calcular métricas para YOLOv8 y YOLO11
    metricas_dataset = calcular_metricas(df, datasets, extra_conditions)

    # Función para convertir métricas en DataFrame
    def metricas_a_dataframe(metricas, datasets):
        data = []
        for dataset in datasets:
            mean = metricas[dataset]["mean"]
            std = metricas[dataset]["std"]
            maximum = metricas[dataset]["max"]
            row = {
                "Dataset": dataset,
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

    # Crear DataFrames finales
    df_dataset = metricas_a_dataframe(metricas_dataset, datasets)

    return df_dataset