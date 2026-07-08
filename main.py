from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conversation_history = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message

    if session_id not in conversation_history:
        conversation_history[session_id] = [
            {"role": "system", "content": """
You are a helpful AI assistant for FAST-NUCES Islamabad campus.
You help students with:
- Fee structure and payment deadlines
- Course registration and semester enrollment
- Exam schedules and grading policy
- Societies and clubs information
- Hostel and transport information
- Faculty and department contacts
- Admission requirements

Be friendly, helpful, and respond in simple English.
If you don't know something specific, advise the student 
to contact the relevant department directly.
Sign off as FAST Assistant.
"""}
        ]

    conversation_history[session_id].append({
        "role": "user",
        "content": user_message
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_history[session_id],
        temperature=0.7,
    )

    assistant_reply = response.choices[0].message.content

    conversation_history[session_id].append({
        "role": "assistant",
        "content": assistant_reply
    })

    return {"reply": assistant_reply}

@app.get("/")
def root():
    return {"message": "FAST-NUCES Chatbot API is running"}