<div align="center">

<img src="https://img.shields.io/badge/Care24-AI%20Chatbot%20%7C%20RAG%20Service-purple?style=for-the-badge&logo=openai&logoColor=white" alt="Care24 RAG Service"/>

# 🤖 Care24 — AI Chatbot & RAG Service

### *Intelligent Medical Assistant powered by RAG, Vector Search & Groq LLM*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20LLaMA%204-FF6B35?style=flat-square&logo=meta)](https://groq.com/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-DC244C?style=flat-square)](https://qdrant.tech/)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB-47A248?style=flat-square&logo=mongodb)](https://www.mongodb.com/)
[![Frontend](https://img.shields.io/badge/Frontend%20Repo-Care24-blue?style=flat-square&logo=github)](https://github.com/chiragdhiman99/Care24-frontend)

</div>

---

## 📌 Overview

This is the **AI-powered chatbot microservice** for [Care24](https://github.com/chiragdhiman99/Care24-frontend) — an elderly care management platform.

The chatbot uses a smart **3-way routing architecture** to understand what the user is asking and respond accordingly:

- 🧠 **Medical questions** → Answered using **RAG** (Retrieval-Augmented Generation) from a **Qdrant vector knowledge base**
- 🏥 **Hospital/Platform questions** → Dynamically queries **MongoDB** using AI-generated filters
- 💬 **General chat** → Handled warmly by the LLM directly

All powered by **Groq's ultra-fast LLaMA models**, with support for **image analysis** and **PDF analysis** too.

---

## 🧠 How It Works — The 3-Way Router

```
User sends a message
        │
        ▼
┌─────────────────────────┐
│   Route Classifier LLM  │  ← Groq llama-3.3-70b decides the route
└─────────────────────────┘
        │
   ┌────┴────┐────────────┐
   ▼         ▼            ▼
 RAG1       RAG2       GENERAL
   │         │            │
   ▼         ▼            ▼
Qdrant    MongoDB      Groq LLM
Vector    AI Query     Direct
Search    Generator    Response
   │         │            │
   └────┬────┘────────────┘
        ▼
  Final LLM Response
  (warm, formatted, multilingual)
```

### Route Details

| Route | Trigger | Data Source | Model Used |
|-------|---------|-------------|------------|
| `RAG1` | Medical questions — symptoms, diseases, medicines, health advice | Qdrant Vector DB (`medical_knowledge` collection) | `llama-3.3-70b-versatile` |
| `RAG2` | Hospital data — doctors, caregivers, bookings, appointments, services | MongoDB (dynamic query generation) | `llama-3.3-70b-versatile` |
| `GENERAL` | Greetings, casual chat, small talk | No external DB | `llama-3.3-70b-versatile` |

---

## ✨ Features

### 💬 Smart `/ask` Endpoint
- **Intelligent routing** — classifies every query before fetching data
- **RAG-based medical answers** — retrieves top-3 relevant chunks from Qdrant vector store
- **AI MongoDB query generation** — converts natural language to MongoDB filters dynamically
- **Chat history support** — last 5 messages included for context-aware responses
- **Multilingual** — replies in the same language the user writes in (Hindi, Hinglish, English)
- **Sensitive data protection** — fields like email, phone, transactionId, razorpayOrderId are never exposed

### 🖼️ Image Analysis `/analyze-image`
- Upload any **medical image URL** with an optional query
- Powered by **LLaMA 4 Scout 17B (multimodal)** via Groq
- Get instant AI analysis — prescriptions, reports, skin conditions, X-rays, etc.
- Returns markdown-formatted, easy-to-read responses

### 📄 PDF Analysis `/analyze-pdf`
- Upload a **PDF file** (medical reports, documents) directly
- Extracts full text using **PyMuPDF (fitz)**
- Sends content + user query to LLaMA 4 Scout for analysis
- Summarizes complex medical documents in simple language

### 🔍 RAG Ingestion Pipeline
- Documents chunked and embedded via custom `ingestion/embedder.py`
- Stored in **Qdrant** as dense vectors under `medical_knowledge` collection
- Query-time: user question is embedded → top-3 similar chunks retrieved → passed as context to LLM

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Framework** | FastAPI 0.115 |
| **LLM Provider** | Groq (`llama-3.3-70b-versatile`, `llama-4-scout-17b-16e-instruct`) |
| **Vector Database** | Qdrant Cloud |
| **Database** | MongoDB (via pymongo) |
| **PDF Parsing** | PyMuPDF (fitz) |
| **Embeddings** | Custom embedder (`ingestion/embedder.py`) |
| **Validation** | Pydantic v2 |
| **HTTP Client** | httpx + requests |
| **Server** | Uvicorn |
| **Logging** | Python logging → `app.log` |

---

## 📁 Project Structure

```
Care24-Rag-service/
├── Data/                    # Raw knowledge base documents (PDFs, text files)
├── ingestion/
│   └── embedder.py          # Embedding logic — converts text to vectors
├── app.py                   # Main FastAPI app — all routes & routing logic
├── requirements.txt         # Python dependencies
├── app.log                  # Runtime logs
├── .python-version          # Python version pin
└── .gitignore
```

---

## 🚀 API Reference

### `POST /ask`
Main chatbot endpoint with intelligent 3-way routing.

**Request Body:**
```json
{
  "query": "Mujhe diabetes ke baare mein batao",
  "chat_history": [
    { "role": "user", "text": "Hello" },
    { "role": "assistant", "text": "Hi! How can I help you?" }
  ],
  "user_id": "64f1a2b3c4d5e6f7a8b9c0d1",
  "user_name": "Ramesh Kumar"
}
```

**Response:**
```json
{
  "answer": "**Diabetes** ek aisi condition hai jisme aapke blood mein sugar level high ho jaata hai...\n\n- **Type 1**: Pancreas insulin nahi banata\n- **Type 2**: Body insulin sahi se use nahi karta\n\n> Please apne doctor se zaroor milein 🙏"
}
```

---

### `POST /analyze-image`
Analyze a medical image by URL.

**Request Body:**
```json
{
  "img_url": "https://example.com/prescription.jpg",
  "query": "Is prescription mein kya medicines hain?"
}
```

**Response:**
```json
{
  "answer": "Is prescription mein ye medicines hain:\n\n1. **Metformin 500mg** - Subah aur raat khana ke baad\n2. **Amlodipine 5mg** - Roz ek tablet..."
}
```

---

### `POST /analyze-pdf`
Upload and analyze a PDF document.

**Form Data:**
- `file` — PDF file (multipart upload)
- `query` *(optional)* — specific question about the document

**Response:**
```json
{
  "answer": "Aapki blood report mein:\n\n- **Haemoglobin**: 11.2 g/dL *(thoda low hai)*\n- **Blood Sugar (Fasting)**: 98 mg/dL *(normal range mein hai)*..."
}
```

---

## ⚙️ Getting Started

### Prerequisites

- Python `>= 3.11`
- [Qdrant Cloud](https://cloud.qdrant.io/) account (or local Qdrant instance)
- [Groq](https://console.groq.com/) API key
- MongoDB URI

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/chiragdhiman99/Care24-Rag-service.git
cd Care24-Rag-service

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory:

```env
# Groq LLM
GROQ_API_KEY=your_groq_api_key

# Qdrant Vector DB
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# MongoDB
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/care24
```

### Running the Service

```bash
# Development
uvicorn app:app --reload --port 8000

# Production
uvicorn app:app --host 0.0.0.0 --port 8000
```

Service runs at **`http://localhost:8000`** 🚀

Interactive API docs available at **`http://localhost:8000/docs`** 📖

---

## 📥 Data Ingestion (Populating Qdrant)

To ingest medical knowledge documents into the Qdrant vector store:

1. Place your documents (PDFs, text files) inside the `Data/` folder
2. Run the ingestion script:

```bash
python ingestion/embedder.py
```

This will:
- Parse and chunk all documents
- Generate embeddings for each chunk
- Upsert vectors into the `medical_knowledge` Qdrant collection

---

## 🔒 Security & Privacy

- Sensitive MongoDB fields (`email`, `phone`, `transactionId`, `razorpayOrderId`, `paymentMethod`) are **always excluded** from responses
- User bookings are always filtered by `userId` — users can only see their own data
- CORS is configured (update `allow_origins` for production)
- All errors are gracefully caught — no internal details leaked to the user

---

## 📊 Logging

All requests and errors are logged to both console and `app.log`:

```
2026-01-01 10:23:45 - INFO - Route decided: RAG1
2026-01-01 10:23:46 - INFO - Qdrant returned 3 results
2026-01-01 10:23:47 - INFO - Response sent successfully
```

---

## 🔗 Related Repositories

| Service | Link |
|---------|------|
| 🖥️ **Frontend** | [Care24-frontend](https://github.com/chiragdhiman99/Care24-frontend) |
| ⚙️ **Backend API** | [Care24-backend](https://github.com/chiragdhiman99/Care24-backend) |
| 🤖 **RAG Service** | You are here |

🌐 **Live App:** [https://care24-frontend-tjrg.vercel.app/](https://care24-frontend-tjrg.vercel.app/)

---

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

Made with ❤️ by [Chirag Dhiman](https://github.com/chiragdhiman99)

⭐ Star this repo if you found it helpful!

</div>
