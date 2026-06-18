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
    
    print("--- Reporte completo ---")
    resultados_mc = [0.42, 0.51, 0.47, 0.50]
    resultados_qmc = [0.44, 0.49, 0.46, 0.48]
    imprimir_reporte(resultados_mc, resultados_qmc)
