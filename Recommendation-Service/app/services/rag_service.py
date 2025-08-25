from langchain.retrievers import MultiQueryRetriever
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.services.vector_store import get_or_create_vector_store
from typing import Dict, Any, List
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

        print("Initializing RAG Service with Pinecone and Groq...")

        # Initialize the LLM with Groq
        self.llm = ChatGroq(
            temperature=settings.TEMPERATURE,
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.LLM_MODEL_NAME,
        )

        # Get the vector store from our dedicated service
        self.vectorstore = get_or_create_vector_store()
        base_retriever = self.vectorstore.as_retriever(search_kwargs={'k': settings.RETRIEVER_K})

        # Initialize the Multi-Query Retriever
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm
        )

        # Define the prompt template
        # In RAGService.__init__

        # Define the prompt template
        self.prompt = PromptTemplate(
            template="""
            You are an expert nutritionist and health advisor, using guidelines from the Sri Lankan Ministry of Health.
            Use the following context from your knowledge base to answer the user's question. The source of each context snippet is provided.
            Provide a detailed, actionable, and personalized diet and lifestyle recommendation in a clear, easy-to-understand format.

            CONTEXT:
            {documents}

            USER'S HEALTH PROFILE:
            {question}

            Based on the user's health profile and the provided context, generate a comprehensive recommendation.
            Focus on specific foods to eat and avoid, meal plans or examples, and scientific reasoning.
            Wherever possible, mention the source of your information (e.g., "According to the Food Based Dietary Guidelines...").

            IMPORTANT: Conclude your entire response with the following disclaimer, exactly as written:
            'Disclaimer: This information is for educational purposes only and is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for personalized health recommendations.'

            RECOMMENDATION:
            """,
            input_variables=["question", "documents"],
        )

        # Create the RAG chain
        self.rag_chain = self.prompt | self.llm | StrOutputParser()

        # A simple chain for cases where retrieval might fail or isn't needed
        self.simple_chain = (
                PromptTemplate.from_template(
                    """
                    You are an expert nutritionist. Based on your general knowledge, provide a detailed diet and lifestyle recommendation for the following health profile: {question}.
                    RECOMMENDATION:
                    """
                )
                | self.llm
                | StrOutputParser()
        )

        self._initialized = True
        print("RAG Service initialized successfully!")

    def _load_and_split_pdf(self, pdf_path: str) -> List[Document]:
        """Loads a PDF and splits it into manageable chunks."""
        try:
            pdf_loader = PyPDFLoader(pdf_path)
            docs = pdf_loader.load()

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            return text_splitter.split_documents(docs)
        except Exception as e:
            print(f"Error loading or splitting PDF at {pdf_path}: {e}")
            raise

    async def add_pdf_to_index(self, pdf_path: str):
        """Loads, splits, and indexes a PDF into Pinecone."""
        print(f"Processing and indexing PDF: {pdf_path}")
        doc_splits = self._load_and_split_pdf(pdf_path)

        # Add documents to the Pinecone index.
        # This will embed the documents and upload them.
        await self.vectorstore.aadd_documents(doc_splits)
        print(f"Successfully indexed {len(doc_splits)} document chunks from {pdf_path}")

    def format_metrics_to_question(self, metrics: Dict[str, Any], additional_info: str = None) -> str:
        """Formats health metrics into a natural language query for the RAG system."""
        question = (
            f"Gender: {metrics['gender']}, Age: {metrics['age']}, "
            f"Total Cholesterol: {metrics['cholesterol']} mg/dL, HDL: {metrics['hdl']} mg/dL, "
            f"LDL: {metrics['ldl']} mg/dL, Triglycerides: {metrics['triglycerides']} mg/dL, "
            f"BMI: {metrics['bmi']}"
        )

        if additional_info:
            question += f". Additional information: {additional_info}"

        return question

    async def get_recommendation(self, metrics: Dict[str, Any], additional_info: str = None) -> str:
        """Gets a diet recommendation using the RAG chain."""
        question = self.format_metrics_to_question(metrics, additional_info)
        print(f"Formatted query for RAG: {question}")

        try:
            # Retrieve relevant documents from Pinecone
            documents = await self.retriever.ainvoke(question)

            if not documents:
                print("Warning: No relevant documents found in Pinecone. Falling back to general knowledge.")
                return await self.simple_chain.ainvoke({"question": question})

            doc_texts = "\n---\n".join([doc.page_content for doc in documents])

            # Invoke the RAG chain with the formatted question and retrieved documents
            answer = await self.rag_chain.ainvoke({"question": question, "documents": doc_texts})
            return answer
        except Exception as e:
            print(f"Error during RAG chain invocation: {e}")
            # Fallback to the simple chain if RAG fails for any reason
            return await self.simple_chain.ainvoke({"question": question})