import requests

# ✅ Define API endpoint
API_URL = "http://127.0.0.1:8000/chat"

print("\n🤖 Chatbot Ready! Type 'exit' to quit.\n")

while True:
    user_input = input("🗣️ You: ").strip()

    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    try:
        # ✅ Send request to FastAPI chatbot
        response = requests.post(API_URL, json={"text": user_input}, timeout=5)

        # ✅ Handle API errors
        if response.status_code == 200:
            print(f"🤖 Bot: {response.json()['response']}")
        else:
            print(f"❌ Error: Unexpected response ({response.status_code})")

    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the chatbot server.")
    except requests.exceptions.Timeout:
        print("❌ Error: Chatbot response took too long.")
    except Exception as e:
        print(f"❌ Error: {e}")
