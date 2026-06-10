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


def main():
    """Ejecuta un ejemplo pequeño del reporte."""
    resultados_mc = [0.42, 0.51, 0.47, 0.50]
    resultados_qmc = [0.44, 0.49, 0.46, 0.48]
    imprimir_reporte(resultados_mc, resultados_qmc)


if __name__ == "__main__":
    main()
