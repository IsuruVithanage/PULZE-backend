from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
import os
from typing import Dict, Any, Optional, List
from langchain_core.documents import Document


class RAGService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        print("Initializing RAG Service...")

        # Get the PDF path
        pdf_path = settings.get_pdf_path()
        self.is_pdf_loaded = False
        self.doc_splits = []

        # Initialize components that don't depend on PDF
        # Define the prompt template
        self.prompt = PromptTemplate(
            template="""
            Question: {question}

            Documents: {documents}

            Provide a detailed diet and lif style changes recommendation based on the health metrics above and the reference documents.
            Focus on specific foods, meal timing, and nutritional guidelines that would help address the person's health concerns.
            Include scientific rationale where appropriate.

            Diet Recommendation:
            """,
            input_variables=["question", "documents"],
        )

        # Initialize the LLM
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=settings.TEMPERATURE,
        )

        # Create a simple chain for when no PDF is available
        self.simple_chain = StrOutputParser() | self.llm | StrOutputParser()

        # Try to load PDF if it exists
        try:
            if os.path.exists(pdf_path):
                self._load_pdf(pdf_path)
                self.is_pdf_loaded = True
            else:
                print(f"Warning: PDF file not found at {pdf_path}. Service will operate in fallback mode.")
        except Exception as e:
            print(f"Error loading PDF: {str(e)}. Service will operate in fallback mode.")

        self._initialized = True
        print("RAG Service initialized successfully!")

    def _load_pdf(self, pdf_path: str):
        """Load the PDF and initialize the vector store"""
        # Load the PDF content
        pdf_loader = PyPDFLoader(pdf_path)
        docs = pdf_loader.load()

        # Initialize a text splitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        # Split the documents into chunks
        self.doc_splits = text_splitter.split_documents(docs)

        # Create embeddings and vector store
        self.vectorstore = SKLearnVectorStore.from_documents(
            documents=self.doc_splits,
            embedding=HuggingFaceEmbeddings(),
        )

        self.retriever = self.vectorstore.as_retriever(k=settings.RETRIEVER_K)

        # Create the RAG chain
        self.rag_chain = self.prompt | self.llm | StrOutputParser()

    def load_pdf_from_file(self, pdf_path: str) -> bool:
        """Load a PDF file and initialize the vector store"""
        try:
            self._load_pdf(pdf_path)
            self.is_pdf_loaded = True
            return True
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")
            return False

    def format_metrics_to_question(self, metrics: Dict[str, Any], additional_info: str = None) -> str:
        """Format the health metrics into a question for the LLM"""
        question = f"My Gender: {metrics['gender']}, Age: {metrics['age']}, "
        question += f"Chol: {metrics['cholesterol']} mg/dL, HDL: {metrics['hdl']} mg/dL, "
        question += f"LDL: {metrics['ldl']} mg/dL, TG {metrics['triglycerides']} mg/dL, "
        question += f"BMI: {metrics['bmi']}"

        if additional_info:
            question += f". Additional information: {additional_info}"

        question += ". Based on these health metrics, what diet should I follow?"
        return question

    async def get_recommendation(self, metrics: Dict[str, Any], additional_info: str = None) -> str:
        """Get a diet recommendation based on health metrics"""
        # Format metrics into a question
        question = self.format_metrics_to_question(metrics, additional_info)

        if not self.is_pdf_loaded:
            # Fallback mode: use the LLM directly with a more detailed prompt
            detailed_prompt = f"""
            You are a health and nutrition expert. A patient with the following metrics is asking for a diet recommendation:

            {question}

            Provide a detailed diet recommendation focusing on:
            1. Foods to include and avoid
            2. Meal timing and frequency
            3. Portion control
            4. Scientific rationale
            5. Potential health benefits

            Based on your expertise, what diet would you recommend?
            """
            return await self.simple_chain.ainvoke(detailed_prompt)

        # PDF is loaded, use RAG approach
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)

        # Extract content from retrieved documents
        doc_texts = "\n".join([doc.page_content for doc in documents])

        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})

        return answer