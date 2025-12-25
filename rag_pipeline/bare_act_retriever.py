import os
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


class BareActRetriever:
    """
    Retrieves the top k relevant sections from FAISS index
    """

    def __init__(
        self,
        faiss_path: str = "faiss_index_legal",
        model: str = "text-embedding-3-small",
    ):
        self.embeddings = OpenAIEmbeddings(model=model)

        # Check if FAISS index exists
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(
                f"❌ FAISS index not found at '{faiss_path}'. "
                f"Please ensure the index files are present in your deployment. "
                f"Run 'python embed_docs.py' locally to create the index, "
                f"then commit the '{faiss_path}' folder to your repository."
            )

        try:
            self.vectorstores = FAISS.load_local(
                faiss_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            raise Exception(
                f"❌ Failed to load FAISS index from '{faiss_path}': {str(e)}\n"
                f"Make sure the index files (index.faiss and index.pkl) are present."
            )

    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, str]]:
        """Retrieve top k relevant documents for the query"""
        try:
            result = self.vectorstores.similarity_search(query, k)

            # Format the documents
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
        except Exception as e:
            raise Exception(f"Error retrieving documents: {str(e)}")
