from fastapi import FastAPI
from qdrant_client import QdrantClient
from ingestion.embedder import embed
from pydantic import BaseModel
import base64
import fitz
from fastapi import UploadFile, File
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import json
from bson import ObjectId
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLLECTION_NAME = "medical_knowledge"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db_mongo = mongo_client["care24"]


class ImageRequest(BaseModel):
    img_url: str
    query: str = ""


@app.post("/analyze-image")
def analyze_image(request: ImageRequest):
    try:
        user_text = request.query
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": request.img_url}},
                        {
                            "type": "text",
                            "text": f"You are Care 24, a warm medical assistant. {user_text} and  Use markdown formatting. Use bullet points, numbered lists, and bold text where needed.",
                        },
                    ],
                }
            ],
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Error in /analyze-image: {str(e)}")
        return {"answer": "Kuch technical problem aa gayi, thodi der baad try karo 🙏"}


@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...), query: str = ""):
    try:
        content = await file.read()
        doc = fitz.open(stream=content, filetype="pdf")

        all_text = ""
        for page in doc:
            all_text += page.get_text()

        user_text = query or "Summarize this document."

        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": f"You are Care 24, a warm medical assistant. {user_text}\n\nDocument content:\n{all_text[:8000]} Use markdown formatting. Use bullet points, numbered lists, and bold text where needed.",
                }
            ],
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Error in /analyze-pdf: {str(e)}")
        return {
            "answer": "PDF process karne mein problem aayi, thodi der baad try karo 🙏"
        }


class QueryRequest(BaseModel):
    query: str
    chat_history: list = []
    user_id: str = ""
    user_name: str = ""


def groq_chat(system_prompt: str, history: list, user_message: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content


@app.post("/ask")
def ask(request: QueryRequest):
    try:
        query = request.query
        history_messages = []

        for msg in request.chat_history[-5:]:
            role = "user" if msg["role"] == "user" else "assistant"
            history_messages.append({"role": role, "content": msg["text"]})

        route_prompt = """You are a routing assistant. Classify the user's question into exactly one of two categories:

RAG1 - if the question is about medical knowledge, symptoms, diseases, medicines, treatments, health advice

RAG2 - if the question is related about hospital data like doctors, nurses, appointments, availability, services, staff and recognize the question if the question seems that its related to website or hospital data so please answer in rag2

GENERAL - only general greetings and casual small talk like hi, hello, how are you, good morning

Reply with ONLY one word: RAG1 or RAG2 or GENERAL. Nothing else."""

        route = groq_chat(route_prompt, history_messages, query).strip()
        logger.info(f"Route decided: {route}")

        if "RAG1" in route:
            route = "RAG1"
        elif "RAG2" in route:
            route = "RAG2"
        elif "GENERAL" in route:
            route = "GENERAL"
        else:
            route = "GENERAL"

        if route == "RAG1":
            try:
                query_vector = embed(query)
                result = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector.tolist(),
                    limit=3,
                ).points
                context = "\n\n".join([r.payload["text"] for r in result])
            except Exception as e:
                logger.error(f"Qdrant error: {str(e)}")
                return {
                    "answer": "Health knowledge base abhi available nahi hai, thodi der baad try karo 🙏"
                }

            rag1_prompt = """You are Care 24, a warm and caring medical assistant for a healthcare platform. Your job is ONLY to help users with health-related questions.

## WHO YOU ARE:
- A friendly, trusted family doctor-like assistant
- You serve mostly elderly users, so be extra gentle and simple
- Never ask about your own wellbeing ("how about you / main theek hoon")
- Always keep focus on the user's health concern

## HOW YOU TALK:
- Simple words, no heavy medical jargon
- Short and clear answers, not too long
- Warm and reassuring tone, never scary
- Never say "Unfortunately", "I cannot", "according to the context"
- Always suggest consulting a real doctor for serious issues

## LANGUAGE RULE:
- Reply in the SAME language the user uses
- English user → reply in English
- Hindi/Hinglish user → reply in Hinglish
- Never switch languages on your own

## important thing:
- Use markdown formatting. Use bullet points, numbered lists, and bold text where needed.
- Always refer to the previous conversation history to understand context before answering.

## CONTEXT RULE:
- If the retrieved context is irrelevant to the question, IGNORE it completely
- Use your own medical knowledge to answer instead
- Never mix unrelated medical topics

## IF USER IS NEGATIVE OR ABUSIVE:
- Politely say: "I'm sorry, I can only help you with health-related questions 😊"
"""
            answer = groq_chat(
                rag1_prompt,
                history_messages,
                f"Context:\n{context}\n\nQuestion: {query}",
            )
            return {"answer": answer}

        elif route == "GENERAL":
            general_prompt = "You are Care24 assistant, a warm and friendly assistant. Answer greetings and general questions simply and warmly. Use markdown formatting. Use bullet points, numbered lists, and bold text where needed. Always refer to the previous conversation history to understand context before answering."
            answer = groq_chat(general_prompt, history_messages, query)
            return {"answer": answer}

        else:
            mongo_prompt = f"""You are a MongoDB query generator.

current user id = {request.user_id}
current user name = {request.user_name}

My database has these collections:

caregivers: {{ _id, name, phone, email, city, experience, specializations, hourlyRate, dailyRate, available, emergencyAvailable, rating, totalReviews, status, topRated, verified }}

bookings: {{ _id, userId, userEmail, caregiverName, caregiverId, patientName, patientAge, patientGender, serviceType, date, startTime, duration, totalAmount, paymentStatus, status, notes }}

Important rules:
- For bookings queries, ALWAYS filter by userId unless user asks for all bookings
- Use exact field names from the schema
- Be as specific as possible in filters

users: {{ _id, name, email, role, phone }}

Return ONLY a JSON object like this:
{{
  "collection": "caregivers",
  "filter": {{ "available": true }}
}}

Nothing else. No explanation. No markdown. No backticks."""

            store = groq_chat(mongo_prompt, history_messages, query).strip()
            logger.info(f"AI RAW RESPONSE: {store}")

            if "```" in store:
                parts = store.split("```")
                store = parts[1]
                if store.lower().startswith("json"):
                    store = store[4:]
                store = store.strip()

            if not store:
                return {"answer": "Invalid query. Please try again."}

            try:
                parsed = json.loads(store)
            except json.JSONDecodeError:
                return {"answer": "Invalid query. Please try again."}

            collection_name = parsed["collection"]
            filter_query = parsed["filter"]

            try:
                mongo_collection = db_mongo[collection_name]
                for key in filter_query:
                    if key == "userId":
                        filter_query[key] = ObjectId(request.user_id)
                    elif key in ["caregiverId", "_id"]:
                        try:
                            filter_query[key] = ObjectId(filter_query[key])
                        except:
                            pass

                mongo_result = list(
                    mongo_collection.find(
                        filter_query,
                        {
                            "_id": 0,
                            "email": 0,
                            "phone": 0,
                            "transactionId": 0,
                            "razorpayOrderId": 0,
                            "paymentMethod": 0,
                            "userEmail": 0,
                        },
                    ).limit(20)
                )
            except Exception as e:
                logger.error(f"MongoDB error: {str(e)}")
                return {
                    "answer": "Database se data fetch karne mein problem aayi, thodi der baad try karo 🙏"
                }

            final_prompt = """You are Care24, a warm and friendly assistant.
User asked a question and database sent the data.
Based on that data give simple and formal message to User.
Use markdown formatting. Use bullet points, numbered lists, and bold text where needed.
Always refer to the previous conversation history to understand context before answering.
NEVER expose sensitive fields like email, phone, password, transactionId, razorpayOrderId, paymentMethod.
Reply in the same language the user used."""

            answer = groq_chat(
                final_prompt,
                history_messages,
                f"User question: {query}\n\nDatabase result: {mongo_result}",
            )
            return {"answer": answer}

    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}")
        return {"answer": "Kuch technical problem aa gayi, thodi der baad try karo 🙏"}
