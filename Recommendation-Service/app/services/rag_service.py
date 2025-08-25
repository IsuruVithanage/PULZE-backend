import os

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
        self.prompt = PromptTemplate(
            template="""
            You are a professional AI health assistant (nutrition & lifestyle).  
            Use ONLY the provided CONTEXT to generate a personalized recommendation for the USER. 
            If required information is missing from the CONTEXT, state: "This information is not available in the provided context."

            ---
            CONTEXT:
            {documents}

            USER PROFILE / QUERY:
            {question}

            ---
            OUTPUT RULES
            - Output must be valid Markdown only. Do NOT use emojis, images, or HTML.
            - Tone: professional, encouraging, non-alarming, and actionable.
            - DO NOT provide medical diagnoses or medication advice.
            - If the CONTEXT does not contain a required food, guideline, or numeric range, write: "Not available in the provided context."
            - Where you reference a guideline or recommendation from CONTEXT, add a short inline citation line under the relevant subsection: `Source: <document id or title from CONTEXT>` (if available).

            ---
            REQUIRED RESPONSE STRUCTURE (use these exact headings and ordering):

            ## Your Health Overview
            Write a **detailed, paragraph-style explanation** of the user’s current health situation.  
            - Discuss each major metric (BMI, cholesterol profile, triglycerides, HDL, LDL, blood sugar, blood pressure, age/gender context) in plain language.  
            - For each metric, include the user’s value, how it compares to the normal range, and what that implies.  
            - Instead of listing values in a rigid table, weave them into **insightful sentences** that connect the dots, e.g.,  
              *“Your BMI of 28 falls above the normal range, suggesting overweight status, which may contribute to cardiovascular risk when combined with slightly elevated triglycerides.”*  
            - Use short **bullets for emphasis** if multiple issues are present, but focus on smooth readability rather than a strict data report.  
            - End this section with a **summary paragraph** giving an overall impression of the user’s current health status and risk level (low, moderate, high), based strictly on CONTEXT.

            ## Your Top Priorities
            - Give **3** prioritized actions (single-line bullets) derived strictly from CONTEXT and the user's highest-priority metrics.

            ## Your Nutrition Plan
            **IMPORTANT:** Use ONLY food items, dietary principles and rationale that appear in the CONTEXT.
            - ### Foods to Emphasize
              - List **5–7** foods/groups with 1-line reasons tied to the CONTEXT. For each item add a `Source:` line if available.
            - ### Foods to Limit
              - List **5–7** foods/groups to reduce/avoid with 1-line reasons tied to the CONTEXT. Add `Source:` lines where possible.

            ## A Day of Healthy Eating
            - Provide one sample day: **Breakfast / Lunch / Dinner / Snack**. Use ONLY food examples and meal patterns present in the CONTEXT. If CONTEXT lacks meal examples, state: "Meal examples not available in context."

            ## Lifestyle Recommendations
            - Actionable bullets for **Physical Activity**, **Sleep**, **Stress Management**, and **Monitoring**.
            - Each bullet must be grounded in CONTEXT; if CONTEXT lacks guidance for any subtopic, state "Not available in the provided context for <subtopic>."

            ---
            FINAL LINE (must be exact):
            **Disclaimer: This information is for educational purposes only and is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for personalized health recommendations.**

            """,
            input_variables=["question", "documents"],
        )

        # Your existing chain will work perfectly with this new prompt
        # self.rag_chain = self.prompt | self.llm | StrOutputParser()
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

    async def reindex_all_sources(self) -> dict:
        """
        Discovers and indexes all PDF documents from the data/sources directory
        and all of its subdirectories.

        Returns:
            A dictionary with the count of indexed files and chunks.
        """
        print("Starting full re-indexing process from all sources...")
        source_directory = os.path.join(settings.PDF_DIRECTORY, "sources")

        if not os.path.isdir(source_directory):
            message = f"Error: Source directory not found at '{source_directory}'"
            print(message)
            raise FileNotFoundError(message)

        indexed_files_count = 0
        total_chunks_count = 0

        # Walk through the sources directory and its subdirectories
        for root, dirs, files in os.walk(source_directory):
            for filename in files:
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(root, filename)
                    try:
                        # Re-using the existing add_pdf_to_index logic
                        # First, load and split to get the chunk count
                        doc_splits = self._load_and_split_pdf(file_path)
                        chunk_count = len(doc_splits)

                        # Then, add to the vector store
                        await self.vectorstore.aadd_documents(doc_splits)

                        print(f"Successfully indexed {chunk_count} chunks from {file_path}")
                        indexed_files_count += 1
                        total_chunks_count += chunk_count

                    except Exception as e:
                        print(f"Failed to index {filename}. Error: {e}")

        summary = {
            "indexed_files": indexed_files_count,
            "total_chunks_indexed": total_chunks_count,
            "message": "Full re-indexing process completed."
        }
        print(summary)
        return summary