"""Herramientas simples para resumir resultados de MC y QMC."""

from __future__ import annotations

import numpy as np


def calcular_estadisticas(valores):
    """Calcula promedio, desviación estándar, mínimo y máximo."""
    datos = np.asarray(valores, dtype=float)
    if datos.size == 0:
        raise ValueError("La lista de valores no puede estar vacía.")

    return {
        "promedio": float(np.mean(datos)),
        "desviacion_estandar": float(np.std(datos)),
        "minimo": float(np.min(datos)),
        "maximo": float(np.max(datos)),
    }


def mean_absolute_error(series_a, series_b):
    """
    Calcula el error absoluto medio entre dos series.
    
    Args:
        series_a: Primera serie (lista o array).
        series_b: Segunda serie (lista o array).
    
    Returns:
        float: Promedio de los valores absolutos de las diferencias.
    
    Raises:
        ValueError: Si las series están vacías o tienen longitudes diferentes.
    """
    datos_a = np.asarray(series_a, dtype=float)
    datos_b = np.asarray(series_b, dtype=float)
    
    if datos_a.size == 0 or datos_b.size == 0:
        raise ValueError("Las series no pueden estar vacías.")
    
    if datos_a.size != datos_b.size:
        raise ValueError("Las series deben tener la misma longitud.")
    
    errores = np.abs(datos_a - datos_b)
    return float(np.mean(errores))


def relative_improvement(classic_error, quantum_error):
    """
    Calcula la mejora relativa del error cuántico respecto al clásico.
    
    Args:
        classic_error: Error del método clásico (float > 0).
        quantum_error: Error del método cuántico (float > 0).
    
    Returns:
        float: Mejora relativa en porcentaje (positivo = mejora, negativo = empeoramiento).
    
    Raises:
        ValueError: Si los errores son negativos, cero o no son números válidos.
    """
    try:
        classic_error = float(classic_error)
        quantum_error = float(quantum_error)
    except (TypeError, ValueError):
        raise ValueError("Los errores deben ser números válidos.")
    
    if classic_error <= 0:
        raise ValueError("El error clásico debe ser mayor que cero.")
    
    if quantum_error < 0:
        raise ValueError("El error cuántico no puede ser negativo.")
    
    mejora = ((classic_error - quantum_error) / classic_error) * 100
    return float(mejora)


def rolling_average(values, window):
    """
    Calcula el promedio móvil sobre una ventana deslizante.
    
    Args:
        values: Lista o array de valores numéricos.
        window: Tamaño de la ventana (int > 0).
    
    Returns:
        list: Lista con los promedios móviles. Los primeros (window - 1) 
              valores no tendrán promedio completo.
    
    Raises:
        ValueError: Si la ventana es inválida o la lista está vacía.
    """
    datos = np.asarray(values, dtype=float)
    
    if datos.size == 0:
        raise ValueError("La lista de valores no puede estar vacía.")
    
    try:
        window = int(window)
    except (TypeError, ValueError):
        raise ValueError("La ventana debe ser un número entero.")
    
    if window <= 0:
        raise ValueError("La ventana debe ser mayor que cero.")
    
    if window > datos.size:
        raise ValueError(f"La ventana ({window}) no puede ser mayor que la cantidad de datos ({datos.size}).")
    
    promedios = []
    for i in range(len(datos) - window + 1):
        promedio_ventana = float(np.mean(datos[i:i + window]))
        promedios.append(promedio_ventana)
    
    return promedios


def analizar_convergencia_cuantica(resultados_mc, resultados_qmc, valor_referencia):
    """
    Analiza la convergencia de los métodos clásico y cuántico hacia un valor de referencia.
    
    Calcula para ambos métodos:
    - Errores acumulativos a medida que procesa más muestras.
    - Velocidad de convergencia comparativa.
    - Punto de convergencia (donde el error se estabiliza bajo un umbral).
    
    Args:
        resultados_mc: Lista de estimaciones del método clásico (float).
        resultados_qmc: Lista de estimaciones del método cuántico (float).
        valor_referencia: Valor verdadero/esperado para comparación (float).
    
    Returns:
        dict: Contiene:
            - errores_acumulados_mc: Errores absolutos acumulados de MC.
            - errores_acumulados_qmc: Errores absolutos acumulados de QMC.
            - promedio_final_mc: Estimación final del método clásico.
            - promedio_final_qmc: Estimación final del método cuántico.
            - convergencia_mejor: Cuál método converge más rápido ('MC', 'QMC' o 'equivalente').
            - punto_convergencia_mc: Índice donde MC se estabiliza (o -1 si no converge).
            - punto_convergencia_qmc: Índice donde QMC se estabiliza (o -1 si no converge).
    
    Raises:
        ValueError: Si las listas tienen diferente longitud o están vacías.
    """
    mc = np.asarray(resultados_mc, dtype=float)
    qmc = np.asarray(resultados_qmc, dtype=float)
    
    if mc.size == 0 or qmc.size == 0:
        raise ValueError("Las listas de resultados no pueden estar vacías.")
    
    if mc.size != qmc.size:
        raise ValueError("Las listas de resultados deben tener la misma longitud.")
    
    valor_ref = float(valor_referencia)
    
    # Calcular errores acumulativos como promedio móvil del error absoluto
    errores_mc = np.abs(mc - valor_ref)
    errores_qmc = np.abs(qmc - valor_ref)
    
    # Usar ventanas acumulativas para ver convergencia
    errores_acumulados_mc = [np.mean(errores_mc[:i+1]) for i in range(len(errores_mc))]
    errores_acumulados_qmc = [np.mean(errores_qmc[:i+1]) for i in range(len(errores_qmc))]
    
    # Determinar punto de convergencia (cuando el error < 5% del valor de referencia)
    umbral_convergencia = abs(valor_ref) * 0.05
    punto_conv_mc = next((i for i, e in enumerate(errores_acumulados_mc) if e < umbral_convergencia), -1)
    punto_conv_qmc = next((i for i, e in enumerate(errores_acumulados_qmc) if e < umbral_convergencia), -1)
    
    # Comparar velocidad de convergencia
    if punto_conv_qmc == -1 and punto_conv_mc == -1:
        convergencia_mejor = "equivalente"
    elif punto_conv_qmc == -1:
        convergencia_mejor = "MC"
    elif punto_conv_mc == -1:
        convergencia_mejor = "QMC"
    else:
        convergencia_mejor = "QMC" if punto_conv_qmc < punto_conv_mc else "MC"
    
    promedio_final_mc = float(np.mean(mc))
    promedio_final_qmc = float(np.mean(qmc))
    
    return {
        "errores_acumulados_mc": errores_acumulados_mc,
        "errores_acumulados_qmc": errores_acumulados_qmc,
        "promedio_final_mc": promedio_final_mc,
        "promedio_final_qmc": promedio_final_qmc,
        "convergencia_mejor": convergencia_mejor,
        "punto_convergencia_mc": punto_conv_mc,
        "punto_convergencia_qmc": punto_conv_qmc,
    }


def imprimir_reporte(resultados_mc, resultados_qmc):
    """Imprime un reporte legible con el resumen de ambos conjuntos."""
    stats_mc = calcular_estadisticas(resultados_mc)
    stats_qmc = calcular_estadisticas(resultados_qmc)
    diferencia_promedios = abs(stats_mc["promedio"] - stats_qmc["promedio"])

    print("Resumen de resultados")
    print("=" * 22)
    print(f"MC  -> promedio: {stats_mc['promedio']:.6f}, desviación estándar: {stats_mc['desviacion_estandar']:.6f}, mínimo: {stats_mc['minimo']:.6f}, máximo: {stats_mc['maximo']:.6f}")
    print(f"QMC -> promedio: {stats_qmc['promedio']:.6f}, desviación estándar: {stats_qmc['desviacion_estandar']:.6f}, mínimo: {stats_qmc['minimo']:.6f}, máximo: {stats_qmc['maximo']:.6f}")
    print(f"Diferencia absoluta entre promedios: {diferencia_promedios:.6f}")


if __name__ == "__main__":
    # Ejemplo: Comparar errores entre métodos clásicos y cuánticos
    print("--- Ejemplo de mean_absolute_error ---")
    serie_1 = [0.45, 0.52, 0.48, 0.50]
    serie_2 = [0.44, 0.49, 0.46, 0.48]
    mae = mean_absolute_error(serie_1, serie_2)
    print(f"Error absoluto medio: {mae:.6f}\n")
    
    print("--- Ejemplo de relative_improvement ---")
    error_clasico = 0.15
    error_cuantico = 0.10
    mejora = relative_improvement(error_clasico, error_cuantico)
    print(f"Mejora relativa: {mejora:.2f}%\n")
    
    print("--- Ejemplo de rolling_average ---")
    datos = [0.41, 0.42, 0.50, 0.51, 0.48, 0.49]
    ventana = 3
    promedios = rolling_average(datos, ventana)
    print(f"Datos originales: {datos}")
    print(f"Promedio móvil (ventana={ventana}): {[f'{p:.2f}' for p in promedios]}\n")
    
    print("--- Análisis de convergencia cuántica ---")
    resultados_mc_conv = [0.50, 0.48, 0.49, 0.495, 0.492, 0.498, 0.501]
    resultados_qmc_conv = [0.45, 0.495, 0.499, 0.5001, 0.5003, 0.4998, 0.5002]
    valor_verdadero = 0.50
    analisis = analizar_convergencia_cuantica(resultados_mc_conv, resultados_qmc_conv, valor_verdadero)
    print(f"Método con mejor convergencia: {analisis['convergencia_mejor']}")
    print(f"Estimación final MC: {analisis['promedio_final_mc']:.6f}")
    print(f"Estimación final QMC: {analisis['promedio_final_qmc']:.6f}")
    print(f"Punto de convergencia MC: índice {analisis['punto_convergencia_mc']}")
    print(f"Punto de convergencia QMC: índice {analisis['punto_convergencia_qmc']}\n")
    
    print("--- Reporte completo ---")
    resultados_mc = [0.42, 0.51, 0.47, 0.50]
    resultados_qmc = [0.44, 0.49, 0.46, 0.48]
    imprimir_reporte(resultados_mc, resultados_qmc)
