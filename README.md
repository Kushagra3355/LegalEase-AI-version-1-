
# ⚖️ LegalEase AI  
**AI-Powered Legal Assistant for Indian Law**

LegalEase AI is an intelligent legal assistant that leverages **RAG (Retrieval-Augmented Generation)**, **FAISS vector search**, and **LangGraph** to help users understand Indian laws and analyze legal documents in simple language.

---

## ✨ Features

- Bare Act–based legal question answering (NyayGPT)
- PDF document upload and Q&A
- FAISS-powered semantic search
- LangGraph-based conversational workflows
- Streaming AI responses
- Streamlit web interface

---

## 📁 Project Structure

```
LegalEase-AI/
│
├── main.py
├── .env
│
├── rag_pipeline/
│   ├── embed_docs.py
│   ├── bare_act_retriever.py
│
├── utils/
│   ├── LegalChatBot.py
│   ├── DocumentQAGraph.py
│
├── faiss_index_legal/
│   ├── index.faiss
│   ├── index.pkl
│
├── legal data/
│   └── *.pdf
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/Kushagra3355/LegalEase-AI.git
cd LegalEase-AI
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

Activate:
- Windows: `venv\Scripts\activate`
- Linux/macOS: `source venv/bin/activate`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 📚 Create FAISS Index

Place Bare Act PDFs in:
```
legal data/
```

Run:
```bash
python rag_pipeline/embed_docs.py
```

---

## 🚀 Run Application
```bash
streamlit run main.py
```

Open:
```
http://localhost:8501
```

---

## 🧠 Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- FAISS
- OpenAI API

---

## 👨‍💻 Author

**Kushagra Omar**  
GitHub: https://github.com/Kushagra3355

---

## 📄 License

MIT License
