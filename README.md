# ⚖️ LegalEase AI

**LegalEase AI** is an AI-powered legal assistant focused on **Indian law**, built using **Streamlit**, **LangChain**, **LangGraph**, **FAISS**, and **OpenAI models**.  
It enables users to ask legal questions, analyze legal documents, and retrieve relevant sections from **Bare Acts**, all with context-aware, explainable responses.

GitHub: https://github.com/Kushagra3355

---

## 📚 Features

- ⚖️ **NyayGPT** – Ask questions about Indian law and legal procedures  
- 📄 **Ask Document** – Upload and analyze legal PDF documents  
- 🔍 **Bare Act Retrieval** using FAISS vector search  
- 💬 **Streaming AI Responses**  
- 📚 Context-aware answers with source references  
- 🧠 Retrieval-Augmented Generation (RAG)  
- 🎨 Clean and simple Streamlit UI  

> ⚠️ **Disclaimer:** This tool is for educational and informational purposes only. It does **not** provide legal advice.

---

## 🏗️ Project Structure

```
LegalEase-AI/
│
├── main.py                     # Streamlit application entry point
├── embed_docs.py               # Optimized Bare Act PDF embedding
├── LegalChatBot.py             # NyayGPT (Legal Q&A system)
├── DocumentQAGraph.py          # Ask Document tool
├── bare_act_retriever.py       # FAISS-based legal retriever
├── faiss_index_legal/          # Generated FAISS index (required)
└── .env                        # OpenAI API key
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Kushagra3355/LegalEase-AI.git
cd LegalEase-AI
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔐 Configuration

### OpenAI API Key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key
```

For **Streamlit Cloud**, add it under **Settings → Secrets**.

---

## 🧠 Creating the FAISS Index (Required)

Before running the app, you **must generate the FAISS vector store**:

```bash
python embed_docs.py
```

This creates the `faiss_index_legal/` directory, which is required at runtime.

> ⚠️ If the index exceeds GitHub size limits, use **Git LFS** or cloud storage.

---

## 🚀 Running the Application

```bash
streamlit run main.py
```

---

## 🧩 Application Modes

### ⚖️ NyayGPT
- Ask questions about Indian laws
- Retrieves relevant Bare Act sections
- Generates concise, easy-to-understand explanations
- Always cites sources
- Never gives legal advice

### 📄 Ask Document
- Upload legal PDFs (judgments, contracts, notices)
- Ask questions based on uploaded documents
- Combines document context with Bare Act references

---

## 🧰 Technologies Used

- **Frontend**: Streamlit  
- **LLM**: OpenAI (GPT-4o-mini)  
- **Embeddings**: text-embedding-3-small  
- **Vector Store**: FAISS  
- **Orchestration**: LangGraph  
- **Backend**: Python  

---

## 🛠 Troubleshooting

**FAISS index not found**
- Run `python embed_docs.py`
- Ensure `faiss_index_legal/` exists

**OpenAI API error**
- Check `.env` file or Streamlit secrets
- Verify API key validity

**Large index size**
- Use Git LFS or external storage

---

## 🚧 Future Enhancements

- Multi-language legal support  
- Case law database integration  
- User authentication  
- Cloud-hosted vector database  
- PDF citation highlighting  

---

## 📄 License

MIT License

---

### 👤 Author
**Kushagra**  
GitHub: https://github.com/Kushagra3355

---

⚖️ *LegalEase AI – Making Indian law more accessible, one question at a time.*
