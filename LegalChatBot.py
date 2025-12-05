from langgraph.graph import StateGraph
from typing import List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from bare_act_retriever import BareActRetriever


class GraphState(TypedDict):
    query: str
    context_docs: List[str]
    messages: List[BaseMessage]
    response: str


class LegalGraphChatBot:
    def __init__(
        self,
        faiss_path: str = "faiss_index_legal",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
    ):
        try:
            self.retriever = BareActRetriever(faiss_path=faiss_path, model=embedding_model)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
        except Exception as e:
            raise Exception(f"Failed to initialize retriever: {str(e)}")
        
        try:
            self.llm = ChatOpenAI(model=llm_model)
        except Exception as e:
            raise Exception(f"Failed to initialize LLM: {str(e)}. Check your OPENAI_API_KEY.")
        
        self.graph = self._build_graph()

    def _memory_node(self, state: GraphState) -> GraphState:
        query = state["query"]
        state["messages"].append(HumanMessage(content=query))
        return state

    def _retriever_node(self, state: GraphState) -> GraphState:
        try:
            docs = self.retriever.retrieve(state["query"])
            state["context_docs"] = [doc["content"] for doc in docs]
        except Exception as e:
            print(f"Retrieval error: {e}")
            state["context_docs"] = []
        return state

    def _llm_node(self, state: GraphState) -> GraphState:
        system_prompt = """You are a legal assistant specialized in Indian law. You help users understand the law simply,  
            based on their question and the context retrieved from bare acts.
            Keep the explanation as small as possible.
            NEVER give legal advice. Always cite the source (act or section)."""
        
        context = "\n\n".join(state["context_docs"]) if state["context_docs"] else "No relevant context found."
        
        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"],
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion:\n{state['query']}"
            ),
        ]

        try:
            response = self.llm.invoke(messages)
            state["messages"].append(AIMessage(content=response.content))
            state["response"] = response.content
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            state["messages"].append(AIMessage(content=error_msg))
            state["response"] = error_msg
        
        return state

    def _llm_node_streaming(self, state: GraphState):
        """Streaming version of the LLM node that yields chunks"""
        system_prompt = """You are a legal assistant specialized in Indian law. You help users understand the law simply,  
            based on their question and the context retrieved from bare acts.
            Keep the explanation as small as possible.
            NEVER give legal advice. Always cite the source (act or section)."""
        
        context = "\n\n".join(state["context_docs"]) if state["context_docs"] else "No relevant context found."
        
        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"],
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion:\n{state['query']}"
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

    def _build_graph(self):
        graph_builder = StateGraph(GraphState)
        graph_builder.add_node("memory", self._memory_node)
        graph_builder.add_node("retrieve", self._retriever_node)
        graph_builder.add_node("llm", self._llm_node)

        graph_builder.set_entry_point("memory")
        graph_builder.add_edge("memory", "retrieve")
        graph_builder.add_edge("retrieve", "llm")
        graph_builder.set_finish_point("llm")

        return graph_builder.compile()

    def init_state(self) -> GraphState:
        return {"query": "", "messages": [], "context_docs": [], "response": ""}

    def invoke(self, state: GraphState, query: str) -> GraphState:
        state["query"] = query
        return self.graph.invoke(state)

    def invoke_streaming(self, state: GraphState, query: str):
        """Streaming version that yields response chunks"""
        state["query"] = query
        state["messages"].append(HumanMessage(content=query))

        # Run retrieval
        try:
            docs = self.retriever.retrieve(state["query"])
            state["context_docs"] = [doc["content"] for doc in docs]
        except Exception as e:
            print(f"Retrieval error: {e}")
            state["context_docs"] = []

        # Stream LLM response
        yield from self._llm_node_streaming(state)
