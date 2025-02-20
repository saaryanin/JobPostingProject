import torch
from transformers import pipeline, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import wikipedia
import difflib

app = FastAPI()

# ✅ Load Job Classification Model ONCE at startup (caching model)
print("🔹 Loading Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
job_classification = pipeline("text-classification", model="bert-base-uncased", device=0 if torch.cuda.is_available() else -1)

# ✅ Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ✅ Conversation Memory
conversation_memory = {}

# ✅ Input Schema
class JobQuery(BaseModel):
    text: str

# ✅ Helper Function: Detect Job-Related Questions
def is_job_question(text):
    keywords = ["is this job legit", "is this a real job", "does this job sound legitimate", "is this job fake"]
    return any(difflib.get_close_matches(text.lower(), keywords, n=1, cutoff=0.7))

# ✅ Helper Function: Wikipedia Search (with caching)
wiki_cache = {}  # ✅ Store previous Wikipedia responses to reduce API calls
def get_wikipedia_summary(query):
    if query in wiki_cache:
        return wiki_cache[query]  # ✅ Return cached result

    try:
        summary = wikipedia.summary(query, sentences=2)
        wiki_cache[query] = summary  # ✅ Cache result
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"🔍 Multiple results found: {', '.join(e.options[:3])}"
    except wikipedia.exceptions.PageError:
        return "❌ I couldn't find an answer to that."

# ✅ Chatbot API Route
@app.post("/chat")
def chat(query: JobQuery):
    user_text = query.text.strip().lower()

    # ✅ Classify job postings
    if is_job_question(user_text):
        classification = job_classification(user_text)[0]
        label = "✅ Real Job" if classification["label"] == "LABEL_0" else "🚨 Fake Job"
        confidence = classification["score"] * 100
        conversation_memory["last_job"] = label  # ✅ Store last job classification
        return {"response": f"{label} ({confidence:.2f}% confidence)"}

    # ✅ Answer "Are you sure?" based on memory
    elif "are you sure" in user_text and "last_job" in conversation_memory:
        return {"response": f"I'm {confidence:.2f}% confident that this is {conversation_memory['last_job']}"}

    # ✅ Wikipedia search for general questions
    else:
        return {"response": get_wikipedia_summary(user_text)}
