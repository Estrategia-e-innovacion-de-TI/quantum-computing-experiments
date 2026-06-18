"""Cambio el docstring."""


def obtener_precio_con_descuento(valor_inicial: float, descuento_pct: float) -> float:
    """Retorna el valor final luego de aplicar un descuento porcentual."""
    if valor_inicial < 0:
        raise ValueError("valor_inicial no puede ser menor que cero")

    if not 0 <= descuento_pct <= 100:
        raise ValueError("descuento_pct debe estar en el rango de 0 a 100")

    descuento = valor_inicial * descuento_pct / 100
    resultado = valor_inicial - descuento

    return round(resultado, 2)


if __name__ == "__main__":
    valor_calculado = obtener_precio_con_descuento(250.0, 15.0)
    print(valor_calculado)
