import os
import re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
import faiss

load_dotenv()


def clean_text(text: str) -> str:
    """Clean and normalize text to reduce redundancy."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters that don't add semantic value
    text = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/]", "", text)
    # Remove page numbers and common footers/headers
    text = re.sub(r"\bPage \d+\b", "", text)
    text = re.sub(r"\b\d+\s*of\s*\d+\b", "", text)
    return text.strip()


def deduplicate_chunks(chunks, similarity_threshold=0.95):
    """Remove near-duplicate chunks based on text similarity."""
    unique_chunks = []
    seen_texts = set()

    for chunk in chunks:
        # Create a normalized version for comparison
        normalized_text = re.sub(r"\s+", " ", chunk.page_content.lower().strip())

        # Skip if we've seen very similar content
        is_duplicate = False
        for seen_text in seen_texts:
            if len(normalized_text) > 0 and len(seen_text) > 0:
                # Simple similarity check based on character overlap
                overlap = len(set(normalized_text) & set(seen_text)) / len(
                    set(normalized_text) | set(seen_text)
                )
                if overlap > similarity_threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            seen_texts.add(normalized_text)
            unique_chunks.append(chunk)

    return unique_chunks


def filter_meaningful_chunks(chunks, min_length=50):
    """Filter out chunks that are too short or don't contain meaningful content."""
    filtered_chunks = []

    for chunk in chunks:
        content = chunk.page_content.strip()

        # Skip if too short
        if len(content) < min_length:
            continue

        # Skip if mostly numbers or special characters
        if len(re.sub(r"[^a-zA-Z]", "", content)) < len(content) * 0.5:
            continue

        # Skip if it's just headers/footers/page numbers
        if re.match(r"^(page \d+|chapter \d+|section \d+)$", content.lower()):
            continue

        filtered_chunks.append(chunk)

    return filtered_chunks


def embed_pdfs_optimized(pdf_dir_path: str, faiss_save_path: str):
    """Optimized PDF embedding with size reduction techniques."""
    print("Loading PDFs with optimization...")

    loader = DirectoryLoader(
        path=pdf_dir_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True,
    )

    docs = loader.lazy_load()

    # Optimized chunking parameters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Reduced from 2000
        chunk_overlap=50,  # Reduced from 100
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    chunks = []
    for doc in docs:
        # Clean the document content
        doc.page_content = clean_text(doc.page_content)

        # Add filename to metadata
        doc.metadata["source"] = os.path.basename(
            doc.metadata.get("source", "unknown.pdf")
        )

        # Split and extend chunks
        doc_chunks = splitter.split_documents([doc])
        chunks.extend(doc_chunks)

    print(f"Initial chunks created: {len(chunks)}")

    # Filter out low-quality chunks
    chunks = filter_meaningful_chunks(chunks)
    print(f"After filtering: {len(chunks)}")

    # Remove duplicates
    chunks = deduplicate_chunks(chunks)
    print(f"After deduplication: {len(chunks)}")

    # Use more efficient embedding model configuration
    embed_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=512,  # Reduced dimensions for smaller file size
        chunk_size=1000,  # Process embeddings in batches
    )

    # Create FAISS index with compression
    vectorstore = FAISS.from_documents(chunks, embedding=embed_model)

    # Optimize the FAISS index for storage
    if hasattr(vectorstore.index, "train") and len(chunks) > 1000:
        # Use IVF (Inverted File) index for better compression on larger datasets
        print("Optimizing FAISS index for storage...")
        nlist = min(100, len(chunks) // 10)  # Number of clusters
        quantizer = faiss.IndexFlatL2(512)  # Using reduced dimensions
        index = faiss.IndexIVFFlat(quantizer, 512, nlist)

        # Train the index if we have enough data
        import numpy as np

        embeddings = np.array(
            [vectorstore.index.reconstruct(i) for i in range(vectorstore.index.ntotal)]
        )
        index.train(embeddings)
        index.add(embeddings)

        # Replace the index
        vectorstore.index = index

    # Save with compression
    vectorstore.save_local(faiss_save_path)

    print(f"Optimized FAISS vector store saved to '{faiss_save_path}'")

    # Print size information
    total_size = 0
    for root, dirs, files in os.walk(faiss_save_path):
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            total_size += size
            print(f"  {file}: {size/1024/1024:.2f} MB")

    print(f"Total index size: {total_size/1024/1024:.2f} MB")


if __name__ == "__main__":
    embed_pdfs_optimized(
        pdf_dir_path="legal data", faiss_save_path="faiss_index_legal_optimized"
    )
