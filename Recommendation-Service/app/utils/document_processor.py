import os
from typing import List

import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain_core.documents import Document

from app.core.config import settings

def _load_and_split_pdf(pdf_path: str, embedding_model, similarity_threshold: float) -> List[Document]:
    """
    Loads and processes a single PDF file for ingestion into the vector store.

    This function uses a robust, multi-stage strategy:
    1. Tries multiple text extraction methods to handle different PDF layouts.
    2. Falls back to Optical Character Recognition (OCR) for scanned documents.
    3. Splits the extracted text into semantically coherent chunks ("Contextual Chunking").
    4. Attaches relevant metadata (source, category) to each chunk.

    Args:
        pdf_path: The local file path to the PDF.
        embedding_model: The sentence-transformer model used for semantic analysis.
        similarity_threshold: The cosine similarity score below which sentences
                              are split into new chunks.

    Returns:
        A list of LangChain Document objects, ready for indexing.
    """
    print(f"--- Starting robust processing for: {os.path.basename(pdf_path)} ---")
    full_text = ""
    MIN_TEXT_LENGTH = 250

    # 1. Attempt robust text extraction using a series of loaders.
    loaders = [
        ("PyMuPDFLoader", PyMuPDFLoader(pdf_path)),
        ("UnstructuredPDFLoader (layout)", UnstructuredPDFLoader(pdf_path, mode="single", strategy="fast")),
    ]
    for loader_name, loader in loaders:
        try:
            print(f"Attempting to load with: {loader_name}...")
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            if len(text) > MIN_TEXT_LENGTH:
                print(f"Success! {loader_name} extracted {len(text)} characters.")
                full_text = text
                break
        except Exception as e:
            print(f"{loader_name} failed: {e}")
            continue

    # 2. If standard loaders fail, fall back to OCR.
    if not full_text:
        print("Warning: Standard loaders failed. Falling back to OCR with Unstructured...")
        try:
            ocr_loader = UnstructuredPDFLoader(pdf_path, mode="single", strategy="hi_res")
            docs = ocr_loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            if len(text) > MIN_TEXT_LENGTH:
                print(f"Success! OCR extracted {len(text)} characters.")
                full_text = text
        except Exception as e:
            print(f"FATAL: OCR loader also failed for {pdf_path}: {e}")
            return []

    # 3. Proceed to the chunking stage.
    print("Proceeding to hybrid chunking stage...")
    try:
        sentences = nltk.sent_tokenize(full_text)
        MIN_SENTENCES_FOR_CONTEXTUAL = 5
        chunk_texts: List[str]

        # Use a simple recursive splitter for very short documents.
        if len(sentences) < MIN_SENTENCES_FOR_CONTEXTUAL:
            print(f"Warning: Only {len(sentences)} sentences found. Using RecursiveCharacterTextSplitter.")
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            chunk_texts = text_splitter.split_text(full_text)
        else:
            # For longer documents, use contextual chunking.
            print(f"Document has {len(sentences)} sentences. Using Contextual Chunker.")
            # Convert each sentence to a numerical vector (embedding).
            sentence_embeddings = embedding_model.embed_documents(sentences)

            # Calculate the semantic similarity between adjacent sentences.
            similarities = []
            for i in range(len(sentence_embeddings) - 1):
                embedding1 = np.array(sentence_embeddings[i]).reshape(1, -1)
                embedding2 = np.array(sentence_embeddings[i + 1]).reshape(1, -1)
                sim = cosine_similarity(embedding1, embedding2)[0][0]
                similarities.append(sim)

            # Group sentences into chunks based on the similarity threshold.
            chunks = []
            current_chunk_sentences = [sentences[0]]
            for i in range(len(similarities)):
                if similarities[i] < similarity_threshold:
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = [sentences[i + 1]]
                else:
                    current_chunk_sentences.append(sentences[i + 1])
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            chunk_texts = chunks

        print(f"Successfully split document into {len(chunk_texts)} chunks.")

        # 4. Extract metadata (category and source) from the file path.
        path_parts = pdf_path.split(os.sep)
        category = "unknown"
        if "sources" in path_parts:
            category_index = path_parts.index("sources") + 1
            if category_index < len(path_parts):
                category = path_parts[category_index]
        source_filename = os.path.basename(pdf_path)

        # 5. Create final Document objects with text and metadata.
        chunk_documents = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={"source": source_filename, "category": category, "chunk_number": i + 1}
            )
            chunk_documents.append(chunk_doc)

        return chunk_documents
    except Exception as e:
        print(f"Error during chunking stage for {pdf_path}: {e}")
        raise