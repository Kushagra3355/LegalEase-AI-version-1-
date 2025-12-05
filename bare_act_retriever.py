from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


class BareActRetriever:
    """
    Retrives the top k relevant sections
    """

    def __init__(
        self,
        faiss_path: str = "faiss_index_legal_optimized",
        model: str = "text-embedding-3-small",
    ):
        self.embeddings = OpenAIEmbeddings(model=model)
        self.vectorstores = FAISS.load_local(
            faiss_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, str]]:
        result = self.vectorstores.similarity_search(query, k)

        # format the doc
        formatted_docs = []
        for i, doc in enumerate(result):
            formatted_docs.append(
                {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "section": doc.metadata.get("section", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                }
            )

        return formatted_docs

