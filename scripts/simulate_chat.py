from oml.app.chat import chat_loop
import threading
import sys
import time

def simulate_input():
    time.sleep(2)
    print("Simulating User Input: Hello, what is RAG?")
    sys.stdin = open("simulated_input.txt", "r")

def run():
    with open("simulated_input.txt", "w") as f:
        f.write("Hello, what is RAG?\nexit\n")
    
    # We patch stdin so chat_loop reads our simulated file
    sys.stdin = open("simulated_input.txt", "r")
    
    try:
        chat_loop(model="mock", top_k=2, budget=1000)
    except Exception as e:
        print(f"Chat simulation ended: {e}")

if __name__ == "__main__":
    run()
