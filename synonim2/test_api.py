from openrouter_client import llamar_modelo


AGENTE_PRUEBA = {
    "modelo": "qwen/qwen3.5-9b",
    "instrucciones": """
Eres un asistente útil.

Reglas:
- Responde siempre en español.
- Explica las cosas de forma clara.
- No inventes información.
""".strip(),
    "temperature": 0.3,
    "max_tokens": 1000,
}


def main() -> None:
    print("Conectado con qwen/qwen3.5-9b")
    print('Escribe "salir" para terminar.\n')

    while True:
        prompt = input("Tú: ").strip()

        if prompt.lower() == "salir":
            print("Programa terminado.")
            break

        if not prompt:
            print("Escribe una pregunta.\n")
            continue

        try:
            respuesta = llamar_modelo(
                agente=AGENTE_PRUEBA,
                mensaje_usuario=prompt,
            )

            print(f"\nQwen: {respuesta}\n")

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()