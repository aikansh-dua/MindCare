"""Simple CLI test for MindCare README stack."""

from chatbot import chat


def main():
    print("MindCare test — type messages (empty line to quit)\n")
    user_id = "test_user"
    samples = [
        "Hello, how are you?",
        "I feel really anxious lately",
        "I can't sleep at night",
    ]
    for msg in samples:
        print(f"You: {msg}")
        result = chat(user_id, msg)
        print(
            f"MindCare: {result['response']}\n"
            f"  emotion={result['emotion']} crisis={result['crisis']} "
            f"memory={result.get('memory_backend')}\n"
        )


if __name__ == "__main__":
    main()
