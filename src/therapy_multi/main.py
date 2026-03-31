#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from therapy_multi.flow import run_turn 


def run():
    history = []
    print("Therapy Multi — Mental Health Chatbot\nType 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() == "quit":
            break

        result = run_turn(user_input, history)

        print(f"\n[Agent: {result['agent_used']}]")
        print(f"Therapist: {result['response']}\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": result["response"]})


if __name__ == "__main__":
    run()