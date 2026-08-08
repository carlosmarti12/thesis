from openrouter_client import call_qwen


def main() -> None:
    answer, usage = call_qwen(
        system_prompt=(
            "You are an expert in English lexical semantics. "
            "Return only the requested answer."
        ),
        user_prompt=(
            "Return five synonyms for 'authentication' "
            "as a JSON array."
        ),
        temperature=0.0,
        max_tokens=100,
    )

    print("Respuesta:")
    print(answer)

    print("\nConsumo:")
    print(usage)


if __name__ == "__main__":
    main()