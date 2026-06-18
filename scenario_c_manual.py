"""Implementación manual posterior a una consulta conceptual."""


def precio_con_rebaja(valor, porcentaje):
    if valor < 0:
        raise ValueError("El valor ingresado debe ser positivo")

    if porcentaje < 0 or porcentaje > 100:
        raise ValueError("El porcentaje debe estar entre 0 y 100")

    rebaja = valor * porcentaje / 100
    total = valor - rebaja

    return round(total, 2)


if __name__ == "__main__":
    print(precio_con_rebaja(250, 15))