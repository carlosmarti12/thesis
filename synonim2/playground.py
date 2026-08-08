import time

from openrouter_client import llamar_modelo

# Edita esto para probar distintos comportamientos.
AGENTE = {
    "modelo": "qwen/qwen3.5-9b",
    "instrucciones": """
Eres un asistente útil.
""".strip(),
    "temperature": 0.3,
}


def main() -> None:
    print(f"Modelo: {AGENTE['modelo']}")
    print('Escribe "salir" para terminar.\n')

    while True:
        prompt = input("Tú: ").strip()

        if prompt.lower() == "salir":
            break

        if not prompt:
            continue

        try:
            inicio = time.perf_counter()
            respuesta = llamar_modelo(agente=AGENTE, mensaje_usuario=prompt)
            duracion = time.perf_counter() - inicio

            print(f"\n[{duracion:.2f}s] {respuesta}\n")

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
