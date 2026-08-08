import time

from openrouter_client import llamar_modelo

AGENTE = {
    "modelo": "qwen/qwen3.5-9b",
    "instrucciones": "Eres un asistente útil. Responde en español, de forma breve.",
    "temperature": 0.3,
}

PROMPT = "Dame 3 sinónimos de la palabra 'rápido'."


def main() -> None:
    inicio = time.perf_counter()
    respuesta = llamar_modelo(agente=AGENTE, mensaje_usuario=PROMPT)
    duracion = time.perf_counter() - inicio

    print(f"Tiempo de respuesta: {duracion:.2f} s\n")
    print("Respuesta:", respuesta)


if __name__ == "__main__":
    main()
