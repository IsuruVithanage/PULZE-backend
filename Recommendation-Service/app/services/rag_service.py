import os
import asyncio

from langchain.retrievers import MultiQueryRetriever
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.models.response_models import StructuredRecommendationResponse, HealthOverview, NutritionPlan, MealPlan, \
    LifestyleRec, TopPriorities, LifestylePlan, RiskAssessment
from app.prompts import prompts
from app.services.vector_store import get_or_create_vector_store, embedding_model
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# -----------------
from langchain.retrievers import MultiQueryRetriever


def _load_and_split_pdf(pdf_path: str, embedding_model, similarity_threshold: float) -> List[Document]:
    """
    Loads a PDF using a robust, multi-stage hybrid strategy. It tries different loaders
    sequentially, including OCR, and then uses a hybrid chunking strategy.
    """
    print(f"--- Starting robust processing for: {os.path.basename(pdf_path)} ---")
    full_text = ""
    MIN_TEXT_LENGTH = 250

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
                break  # Exit the loop on first success
        except Exception as e:
            print(f"{loader_name} failed: {e}")
            continue

    # --- STAGE 2: OCR FALLBACK ---
    # If both standard loaders failed, attempt OCR as a last resort.
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
            return []  # If all loaders fail, we must skip this file.

    # --- STAGE 3: HYBRID CHUNKING ---
    # Now we are guaranteed to have text, we apply our hybrid chunking.
    print("Proceeding to hybrid chunking stage...")
    try:
        sentences = nltk.sent_tokenize(full_text)
        MIN_SENTENCES_FOR_CONTEXTUAL = 5

        chunk_texts: List[str]

        if len(sentences) < MIN_SENTENCES_FOR_CONTEXTUAL:
            print(f"Warning: Only {len(sentences)} sentences found. Using RecursiveCharacterTextSplitter.")
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            chunk_texts = text_splitter.split_text(full_text)
        else:
            print(f"Document has {len(sentences)} sentences. Using Contextual Chunker.")
            # Your full contextual chunking logic
            sentence_embeddings = embedding_model.embed_documents(sentences)
            similarities = []
            for i in range(len(sentence_embeddings) - 1):
                embedding1 = np.array(sentence_embeddings[i]).reshape(1, -1)
                embedding2 = np.array(sentence_embeddings[i + 1]).reshape(1, -1)
                sim = cosine_similarity(embedding1, embedding2)[0][0]
                similarities.append(sim)

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

        # --- STAGE 4: METADATA and DOCUMENT CREATION ---
        path_parts = pdf_path.split(os.sep)
        category = "unknown"
        if "sources" in path_parts:
            category_index = path_parts.index("sources") + 1
            if category_index < len(path_parts):
                category = path_parts[category_index]
        source_filename = os.path.basename(pdf_path)

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
    Translates raw health metric values into qualitative clinical interpretations
    based on the provided guidelines.
    """
    interpretations = {}

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


# --- REPLACE the old function with this new, smarter version ---
def format_metrics_to_question(metrics: Dict[str, Any], additional_info: str = None) -> str:
    """
    Formats health metrics into a rich, interpretive, natural language query for the RAG system.
    """
    # First, get the clinical interpretations of the raw numbers
    interpretations = _interpret_metrics(metrics)

    # Build a narrative query
    query_parts = [
        f"The user is a {metrics.get('age')}-year-old {metrics.get('gender', 'person').lower()}."
    ]

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

    if additional_info:
        query_parts.append(f"Additional user information: {additional_info}")

    # The final query is now a descriptive paragraph
    return " ".join(query_parts)


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

        print("Initializing RAG Orchestration Service...")
        self.llm = ChatGroq(temperature=settings.TEMPERATURE, groq_api_key=settings.GROQ_API_KEY,
                            model_name=settings.LLM_MODEL_NAME)
        self.vectorstore = get_or_create_vector_store()
        base_retriever = self.vectorstore.as_retriever(search_kwargs={'k': settings.RETRIEVER_K})
        self.retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=self.llm)
        self.embedding_model = embedding_model

        # --- UPDATE: Use the new wrapper models for list-based tasks ---
        self.overview_chain = self._create_chain(prompts.OVERVIEW_PROMPT_TEMPLATE, HealthOverview)
        self.priorities_chain = self._create_chain(prompts.PRIORITIES_PROMPT_TEMPLATE, TopPriorities)  # Changed
        self.nutrition_chain = self._create_chain(prompts.NUTRITION_PROMPT_TEMPLATE, NutritionPlan)
        self.meal_plan_chain = self._create_chain(prompts.MEAL_PLAN_PROMPT_TEMPLATE, MealPlan)
        self.lifestyle_chain = self._create_chain(prompts.LIFESTYLE_PROMPT_TEMPLATE, LifestylePlan)
        self.risk_assessment_chain = self._create_chain(prompts.RISK_ASSESSMENT_PROMPT_TEMPLATE, RiskAssessment)# Changed

        self._initialized = True
        print("RAG Orchestration Service initialized successfully!")

    # --- SIMPLIFIED HELPER FUNCTION ---
    def _create_chain(self, prompt_template: str, pydantic_object: Any) -> Runnable:
        """Helper to create a LangChain runnable for a specific Pydantic model."""
        parser = PydanticOutputParser(pydantic_object=pydantic_object)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "documents"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return prompt | self.llm | parser

    async def _generate_task(
            self, chain: Runnable, question: str, metadata_filter: Optional[Dict] = None
    ):
        """
        Helper to run a full RAG chain for a specific task, including retrieval.
        """
        try:
            print(f"Retrieving documents for task with query: '{question[:50]}...'")

            # 1. Perform targeted retrieval for this specific task
            if metadata_filter:
                # Use the filter if provided (for specialist tasks)
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={'k': settings.RETRIEVER_K, 'filter': metadata_filter}
                )
            else:
                # Use the general retriever (for the overview task)
                retriever = self.retriever

            documents = await retriever.ainvoke(question)
            doc_texts = "\n---\n".join([doc.page_content for doc in documents])

            # 2. Invoke the LLM chain with the tailored documents
            print(f"Generating response for task...")
            return await chain.ainvoke({"question": question, "documents": doc_texts})
        except Exception as e:
            print(f"Error in generating task for chain {chain}: {e}")
            return None

    # --- REWRITTEN: The main orchestration method ---
    async def generate_structured_recommendation(
            self, metrics: Dict[str, Any], additional_info: str = None
    ) -> StructuredRecommendationResponse:
        """
        Orchestrates multiple, parallel RAG calls with per-task retrieval to build a recommendation.
        """
        initial_question = format_metrics_to_question(metrics, additional_info)

        # Create specialized questions for each agent to guide their retrieval
        specialized_questions = {
            "overview": initial_question,
            "priorities": f"Based on the following profile, what are the top 3 health priorities? Profile: {initial_question}",
            "nutrition": f"Generate a detailed nutrition plan for a person with this profile: {initial_question}",
            "meal_plan": f"Create a Sri Lankan meal plan for a person with this profile: {initial_question}",
            "lifestyle": f"What are the key lifestyle recommendations for this profile? Profile: {initial_question}",
            "risk": f"Provide a final risk assessment and next steps for this profile: {initial_question}"
        }

        # Define the concurrent tasks, each with its own query and metadata filter
        tasks = [
            # The overview agent gets to see everything
            self._generate_task(self.overview_chain, specialized_questions["overview"]),
            # Specialist agents get filtered documents
            self._generate_task(self.priorities_chain, specialized_questions["priorities"], {"category": "general"}),
            self._generate_task(self.nutrition_chain, specialized_questions["nutrition"], {"category": "dietary"}),
            self._generate_task(self.meal_plan_chain, specialized_questions["meal_plan"], {"category": "dietary"}),
            self._generate_task(self.lifestyle_chain, specialized_questions["lifestyle"], {"category": "general"}),
            self._generate_task(self.risk_assessment_chain, specialized_questions["risk"],
                                {"category": "risk_assessment"}),
        ]

        results = await asyncio.gather(*tasks)
        overview_result, priorities_result, nutrition_result, meal_plan_result, lifestyle_result, risk_assessment_result = results

        # Assemble the final response
        final_priorities = priorities_result.priorities if priorities_result else None
        final_lifestyle = lifestyle_result.recommendations if lifestyle_result else None

        return StructuredRecommendationResponse(
            healthOverview=overview_result,
            topPriorities=final_priorities,
            nutritionPlan=nutrition_result,
            mealPlan=meal_plan_result,
            lifestyle=final_lifestyle,
            riskAssessment=risk_assessment_result,
        )

    async def add_pdf_to_index(self, pdf_path: str):
        """Loads, splits, and indexes a PDF into Pinecone using contextual chunking."""
        print(f"Processing and indexing PDF: {pdf_path}")
        doc_splits = _load_and_split_pdf(
            pdf_path,
            self.embedding_model,
            settings.CONTEXTUAL_CHUNK_SIMILARITY_THRESHOLD
        )

        if not doc_splits:
            print(f"Skipping indexing for {pdf_path} as no chunks were generated.")
            return

        await self.vectorstore.aadd_documents(doc_splits)
        print(f"Successfully indexed {len(doc_splits)} contextual chunks from {pdf_path}")

    # In app/services/rag_service.py within the RAGService class

    async def get_recommendation(self, metrics: Dict[str, Any], additional_info: str = None) -> str:
        """Gets a diet recommendation using the RAG chain."""
        question = format_metrics_to_question(metrics, additional_info)
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
            print("ANswerrrrrrr")
            print(answer)
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
                        doc_splits = _load_and_split_pdf(
                            file_path,
                            self.embedding_model,
                            settings.CONTEXTUAL_CHUNK_SIMILARITY_THRESHOLD
                        )

                        if not doc_splits:
                            print(f"Skipping indexing for {filename} as no chunks were generated.")
                            continue

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
            "message": "Full re-indexing process with contextual chunking completed."
        }
        print(summary)
        return summary