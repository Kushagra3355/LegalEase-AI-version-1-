from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from bare_act_retriever import BareActRetriever
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    query: str
    response: str
    messages: List[BaseMessage]
    context_docs: List[str]
    context_legal: List[str]


class DocumentQATool:
    def __init__(
        self,
        faiss_path: str = "faiss_index_legal",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
    ):
        self.embedding_model = embedding_model
        self.retriever = BareActRetriever(faiss_path=faiss_path, model=embedding_model)
        self.llm = ChatOpenAI(model=llm_model)
        self.graph = self.build_graph()
        self.vectorstore: FAISS = None  # Initialized later in upload_pdf_and_embed()

    def upload_pdf_and_embed(self, pdf_path: str, state: GraphState) -> bool:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
            chunks = splitter.split_documents(docs)
            self.vectorstore = FAISS.from_documents(
                chunks, embedding=OpenAIEmbeddings(model=self.embedding_model)
            )
            return True
        except Exception as e:
            print(f"Failed to upload and embed PDF: {e}")
            return False

    def _retriever_node_legal_data(self, state: GraphState) -> GraphState:
        docs = self.retriever.retrieve(state["query"])
        try:
            state["context_legal"] = [doc.page_content for doc in docs]
        except AttributeError:
            state["context_legal"] = [doc["content"] for doc in docs]
        return state

    def _retriever_node_docs(self, state: GraphState) -> GraphState:
        if not self.vectorstore:
            print("Warning: Vectorstore is not initialized. Skipping doc retrieval.")
            state["context_docs"] = []
            return state
        docs = self.vectorstore.similarity_search(state["query"], k=4)
        state["context_docs"] = [doc.page_content for doc in docs]
        return state

    def _memory_node(self, state: GraphState) -> GraphState:
        query = state["query"]
        state["messages"].append(HumanMessage(content=query))
        return state

    def llm_node(self, state: GraphState) -> GraphState:
        system_prompt = """You are a legal assistant specialized in Indian law. 
You help users understand legal judgments and official documents in simple, clear language.
You refer to Bare Act data and retrieved documents to answer the question.
NEVER give legal advice. Always cite the Act or Section where possible."""

        context_docs = "\n\n".join(state["context_docs"])
        context_legal = "\n\n".join(state["context_legal"])

        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"],
            HumanMessage(
                content=f"Context:\n{context_docs}\n\nQuestion:\n{state['query']}\n\nBare Acts:\n{context_legal}"
            ),
        ]

        response = self.llm.invoke(messages)
        state["messages"].append(AIMessage(content=response.content))
        state["response"] = response.content
        return state

    def llm_node_streaming(self, state: GraphState):
        """Streaming version of the LLM node that yields chunks"""
        system_prompt = """You are a legal assistant specialized in Indian law. 
You help users understand legal judgments and official documents in simple, clear language.
You refer to Bare Act data and retrieved documents to answer the question.
NEVER give legal advice. Always cite the Act or Section where possible."""

        context_docs = "\n\n".join(state["context_docs"])
        context_legal = "\n\n".join(state["context_legal"])

        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"],
            HumanMessage(
                content=f"Context:\n{context_docs}\n\nQuestion:\n{state['query']}\n\nBare Acts:\n{context_legal}"
            ),
        ]

        full_response = ""
        try:
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            full_response = error_msg
            yield error_msg

        state["messages"].append(AIMessage(content=full_response))
        state["response"] = full_response

    def build_graph(self):
        builder = StateGraph(GraphState)

        builder.add_node("memory", self._memory_node)
        builder.add_node("retriever_legal", self._retriever_node_legal_data)
        builder.add_node("retriever_doc", self._retriever_node_docs)
        builder.add_node("llm", self.llm_node)

        builder.set_entry_point("memory")
        builder.add_edge("memory", "retriever_legal")
        builder.add_edge("retriever_legal", "retriever_doc")
        builder.add_edge("retriever_doc", "llm")
        builder.set_finish_point("llm")

        return builder.compile()

    def init_state(self) -> GraphState:
        return {
            "query": "",
            "messages": [],
            "context_docs": [],
            "context_legal": [],
            "response": "",
        }

    def invoke(self, state: GraphState, query: str) -> GraphState:
        state["query"] = query
        return self.graph.invoke(state)

    def invoke_streaming(self, state: GraphState, query: str):
        """Streaming version that yields response chunks"""
        state["query"] = query
        state["messages"].append(HumanMessage(content=query))

        # Run retrieval for legal data
        docs = self.retriever.retrieve(state["query"])
        try:
            state["context_legal"] = [doc.page_content for doc in docs]
        except AttributeError:
            state["context_legal"] = [doc["content"] for doc in docs]

        # Run retrieval for documents
        if self.vectorstore:
            docs = self.vectorstore.similarity_search(state["query"], k=4)
            state["context_docs"] = [doc.page_content for doc in docs]
        else:
            state["context_docs"] = []

        # Stream LLM response
        yield from self.llm_node_streaming(state)
