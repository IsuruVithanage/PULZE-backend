"""
Core service for the Retrieval-Augmented Generation (RAG) system.

This module contains the central RAGService class, which orchestrates the entire
process of generating health recommendations. It handles two primary workflows:
1. Ingestion: Processing and indexing PDF documents into a vector store.
2. Generation: Creating structured, AI-powered responses based on user data
   and the indexed knowledge base.
"""
import os
import asyncio
from typing import Dict, Any, List, Optional

import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from app.core.config import settings
from app.models.response_models import StructuredRecommendationResponse, HealthOverview, NutritionPlan, MealPlan, \
    TopPriorities, LifestylePlan, RiskAssessment
from app.prompts import prompts
from app.services.vector_store import get_or_create_vector_store, embedding_model


# --- Private Helper Functions ---
# These functions support the RAGService class but are kept separate for clarity
# and modularity.

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


def _interpret_metrics(metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Translates raw health metric values into qualitative clinical interpretations.
    """
    interpretations = {}
    # (This function contains the business logic for clinical ranges)
    # BMI Interpretation
    if bmi := metrics.get("bmi"):
        if bmi >= 30:
            interpretations['bmi_status'] = "Obese"
        elif bmi >= 25:
            interpretations['bmi_status'] = "Overweight"
        elif bmi >= 23:
            interpretations['bmi_status'] = "At Risk"
        elif bmi >= 18.5:
            interpretations['bmi_status'] = "Normal"
        else:
            interpretations['bmi_status'] = "Underweight"
    # LDL Cholesterol Interpretation
    if ldl := metrics.get("ldl"):
        if ldl >= 190:
            interpretations['ldl_status'] = "Very High"
        elif ldl >= 160:
            interpretations['ldl_status'] = "High"
        elif ldl >= 130:
            interpretations['ldl_status'] = "Borderline High"
        elif ldl >= 100:
            interpretations['ldl_status'] = "Near Optimal"
        else:
            interpretations['ldl_status'] = "Optimal"
    # HDL Cholesterol Interpretation
    if hdl := metrics.get("hdl"):
        if hdl < 35:
            interpretations['hdl_status'] = "Critically Low"
        elif hdl < 40:
            interpretations['hdl_status'] = "Low (Risk Factor)"
        elif hdl >= 60:
            interpretations['hdl_status'] = "High (Protective)"
        else:
            interpretations['hdl_status'] = "Normal"
    # Triglycerides Interpretation
    if tg := metrics.get("triglycerides"):
        if tg >= 500:
            interpretations['tg_status'] = "Very High"
        elif tg >= 200:
            interpretations['tg_status'] = "High"
        elif tg >= 150:
            interpretations['tg_status'] = "Borderline High"
        else:
            interpretations['tg_status'] = "Normal"
    # Fasting Blood Sugar Interpretation
    if fbs := metrics.get("fasting_blood_sugar"):
        if fbs >= 126:
            interpretations['fbs_status'] = "in the Diabetic Range"
        elif fbs >= 100:
            interpretations['fbs_status'] = "in the Pre-diabetic Range"
        else:
            interpretations['fbs_status'] = "Normal"
    return interpretations


def format_metrics_to_question(
        metrics: Dict[str, Any],
        reported_conditions: Optional[List[str]] = None,
        reported_habits: Optional[List[str]] = None,
        additional_info: str = None
) -> str:
    """
    Formats all user health data into a rich, natural language query for the LLM.
    This is a key prompt engineering step.
    """
    interpretations = _interpret_metrics(metrics)
    # Build a descriptive paragraph from the user's data.
    query_parts = [f"The user is a {metrics.get('age')}-year-old {metrics.get('gender', 'person').lower()}."]
    if bmi_status := interpretations.get('bmi_status'):
        query_parts.append(f"Their BMI is {metrics.get('bmi')} kg/m², which is in the '{bmi_status}' category.")
    lab_results = []
    if ldl_status := interpretations.get('ldl_status'):
        lab_results.append(f"LDL cholesterol is {ldl_status} at {metrics.get('ldl')} mg/dL")
    if hdl_status := interpretations.get('hdl_status'):
        lab_results.append(f"HDL cholesterol is {hdl_status} at {metrics.get('hdl')} mg/dL")
    if tg_status := interpretations.get('tg_status'):
        lab_results.append(f"triglycerides are {tg_status} at {metrics.get('triglycerides')} mg/dL")
    if fbs_status := interpretations.get('fbs_status'):
        lab_results.append(f"fasting blood sugar is {fbs_status} at {metrics.get('fasting_blood_sugar')} mg/dL")
    if lab_results:
        query_parts.append(f"Key lab results indicate: {', '.join(lab_results)}.")
    if reported_conditions:
        conditions_str = ', '.join(reported_conditions)
        query_parts.append(f"The user also reports the following health conditions: {conditions_str}.")
    if reported_habits:
        habits_str = ', '.join(reported_habits)
        query_parts.append(f"Regarding lifestyle, the user mentions: {habits_str}.")
    if additional_info:
        query_parts.append(f"Additional user information: {additional_info}")
    return " ".join(query_parts)


# --- Main Service Class ---

class RAGService:
    """
    The central orchestration service for the RAG system.
    This class is implemented as a Singleton to ensure only one instance
    is created, which efficiently manages connections to the LLM and vector store.
    """
    _instance = None

    # Singleton pattern: Ensures only one RAGService object is ever created.
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initializes the RAGService, setting up connections to the LLM,
        the vector store, and creating the specialized LangChain chains.
        """
        if self._initialized:
            return

        print("Initializing RAG Orchestration Service...")
        # 1. Initialize the Large Language Model (LLM).
        self.llm = ChatGroq(temperature=settings.TEMPERATURE, groq_api_key=settings.GROQ_API_KEY,
                            model_name=settings.LLM_MODEL_NAME)
        # 2. Initialize the vector store and retriever.
        self.vectorstore = get_or_create_vector_store()
        base_retriever = self.vectorstore.as_retriever(search_kwargs={'k': settings.RETRIEVER_K})
        self.retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=self.llm)
        self.embedding_model = embedding_model

        # 3. Create a specialized, self-correcting chain for each section of the report.
        self.overview_chain = self._create_chain(prompts.OVERVIEW_PROMPT_TEMPLATE, HealthOverview)
        self.priorities_chain = self._create_chain(prompts.PRIORITIES_PROMPT_TEMPLATE, TopPriorities)
        self.nutrition_chain = self._create_chain(prompts.NUTRITION_PROMPT_TEMPLATE, NutritionPlan)
        self.meal_plan_chain = self._create_chain(prompts.MEAL_PLAN_PROMPT_TEMPLATE, MealPlan)
        self.lifestyle_chain = self._create_chain(prompts.LIFESTYLE_PROMPT_TEMPLATE, LifestylePlan)
        self.risk_assessment_chain = self._create_chain(prompts.RISK_ASSESSMENT_PROMPT_TEMPLATE, RiskAssessment)

        self._initialized = True
        print("RAG Orchestration Service initialized successfully!")

    def _create_chain(self, prompt_template: str, pydantic_object: Any) -> Runnable:
        """
        Private helper to build a reliable, self-correcting LangChain runnable.
        Each chain is designed to output a specific, validated JSON structure.
        """
        # The Pydantic parser enforces the desired JSON output schema.
        pydantic_parser = PydanticOutputParser(pydantic_object=pydantic_object)

        # The OutputFixingParser acts as a safety net, using the LLM to fix its own errors.
        auto_fixing_parser = OutputFixingParser.from_llm(
            parser=pydantic_parser,
            llm=self.llm
        )

        # The prompt template includes the user query, retrieved documents, and format instructions.
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "documents"],
            partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
        )

        # The final chain pipes the steps together: Prompt -> LLM -> Parser.
        return prompt | self.llm | auto_fixing_parser

    async def _generate_task(
            self, chain: Runnable, question: str, metadata_filter: Optional[Dict] = None
    ) -> Any:
        """

        Private helper to run a single RAG task asynchronously. It retrieves
        relevant documents (with optional filtering) and then invokes the
        specified chain to generate a structured output.
        """
        try:
            print(f"Retrieving documents for task with query: '{question[:50]}...'")

            # Use a specific retriever if a metadata filter is provided.
            if metadata_filter:
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={'k': settings.RETRIEVER_K, 'filter': metadata_filter}
                )
            else:
                retriever = self.retriever

            # Retrieve relevant documents from the vector store.
            documents = await retriever.ainvoke(question)
            doc_texts = "\n---\n".join([doc.page_content for doc in documents])

            # Invoke the chain with the question and retrieved context.
            print(f"Generating response for task...")
            return await chain.ainvoke({"question": question, "documents": doc_texts})
        except Exception as e:
            # Return None if a task fails to avoid crashing the entire process.
            print(f"Error in generating task for chain {chain}: {e}")
            return None

    async def generate_structured_recommendation(
            self,
            metrics: Dict[str, Any],
            reported_conditions: Optional[List[str]],
            reported_habits: Optional[List[str]],
            additional_info: str = None
    ) -> StructuredRecommendationResponse:
        """
        The main orchestration method for generating a full health report.
        It runs multiple, parallel RAG calls, each tailored to a specific
        section of the report, and then assembles the final structured response.
        """
        # 1. Format the user's raw data into a rich, narrative query.
        initial_question = format_metrics_to_question(
            metrics,
            reported_conditions,
            reported_habits,
            additional_info
        )

        # 2. Create specialized sub-queries for each report section.
        specialized_questions = {
            "overview": initial_question,
            "priorities": f"Based on the following profile, what are the top 3 health priorities? Profile: {initial_question}",
            "nutrition": f"Generate a detailed nutrition plan for a person with this profile: {initial_question}",
            "meal_plan": f"Create a Sri Lankan meal plan for a person with this profile: {initial_question}",
            "lifestyle": f"What are the key lifestyle recommendations for this profile? Profile: {initial_question}",
            "risk": f"Provide a final risk assessment and next steps for this profile: {initial_question}"
        }

        # 3. Prepare a list of asynchronous tasks to be run concurrently.
        # Note the use of metadata filters to retrieve highly relevant context for each task.
        tasks = [
            self._generate_task(self.overview_chain, specialized_questions["overview"]),
            self._generate_task(self.priorities_chain, specialized_questions["priorities"], {"category": "general"}),
            self._generate_task(self.nutrition_chain, specialized_questions["nutrition"], {"category": "dietary"}),
            self._generate_task(self.meal_plan_chain, specialized_questions["meal_plan"], {"category": "dietary"}),
            self._generate_task(self.lifestyle_chain, specialized_questions["lifestyle"], {"category": "general"}),
            self._generate_task(self.risk_assessment_chain, specialized_questions["risk"],
                                {"category": "risk_assessment"}),
        ]

        # 4. Execute all tasks in parallel for maximum efficiency.
        results = await asyncio.gather(*tasks)

        # 5. Unpack the results into named variables.
        overview_result, priorities_result, nutrition_result, meal_plan_result, lifestyle_result, risk_assessment_result = results

        # 6. Safely extract nested data, handling cases where a task might have failed.
        final_priorities = priorities_result.priorities if priorities_result else None
        final_lifestyle = lifestyle_result.recommendations if lifestyle_result else None

        # 7. Assemble the final, structured Pydantic response object.
        return StructuredRecommendationResponse(
            healthOverview=overview_result,
            topPriorities=final_priorities,
            nutritionPlan=nutrition_result,
            mealPlan=meal_plan_result,
            lifestyle=final_lifestyle,
            riskAssessment=risk_assessment_result,
        )

    async def add_pdf_to_index(self, pdf_path: str):
        """Public method to process and index a single PDF into the vector store."""
        print(f"Processing and indexing PDF: {pdf_path}")
        # Call the helper function to load, split, and contextualize the PDF.
        doc_splits = _load_and_split_pdf(
            pdf_path,
            self.embedding_model,
            settings.CONTEXTUAL_CHUNK_SIMILARITY_THRESHOLD
        )

        if not doc_splits:
            print(f"Skipping indexing for {pdf_path} as no chunks were generated.")
            return

        # Add the processed document chunks to the Pinecone vector store.
        await self.vectorstore.aadd_documents(doc_splits)
        print(f"Successfully indexed {len(doc_splits)} contextual chunks from {pdf_path}")

    async def reindex_all_sources(self) -> dict:
        """
        Public method to discover and index all PDFs in the configured source directory.
        """
        print("Starting full re-indexing process from all sources...")
        source_directory = os.path.join(settings.PDF_DIRECTORY, "sources")

        if not os.path.isdir(source_directory):
            message = f"Error: Source directory not found at '{source_directory}'"
            print(message)
            raise FileNotFoundError(message)

        indexed_files_count = 0
        total_chunks_count = 0

        # Walk through all subdirectories and find PDF files.
        for root, dirs, files in os.walk(source_directory):
            for filename in files:
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(root, filename)
                    try:
                        # Process each PDF found.
                        doc_splits = _load_and_split_pdf(
                            file_path,
                            self.embedding_model,
                            settings.CONTEXTUAL_CHUNK_SIMILARITY_THRESHOLD
                        )

                        if not doc_splits:
                            print(f"Skipping indexing for {filename} as no chunks were generated.")
                            continue

                        # Add the chunks to the vector store.
                        chunk_count = len(doc_splits)
                        await self.vectorstore.aadd_documents(doc_splits)
                        print(f"Successfully indexed {chunk_count} chunks from {file_path}")
                        indexed_files_count += 1
                        total_chunks_count += chunk_count
                    except Exception as e:
                        print(f"Failed to index {filename}. Error: {e}")

        # Return a summary of the re-indexing process.
        summary = {
            "indexed_files": indexed_files_count,
            "total_chunks_indexed": total_chunks_count,
            "message": "Full re-indexing process with contextual chunking completed."
        }
        print(summary)
        return summary