import os
import asyncio

from langchain.retrievers import MultiQueryRetriever
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.models.response_models import StructuredRecommendationResponse, HealthOverview, NutritionPlan, MealPlan, \
    LifestyleRec, TopPriorities, LifestylePlan
from app.prompts import prompts
from app.services.vector_store import get_or_create_vector_store, embedding_model
from typing import Dict, Any, List
from langchain_core.documents import Document
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# -----------------
from langchain.retrievers import MultiQueryRetriever


def _load_and_split_pdf(pdf_path: str, embedding_model, similarity_threshold: float) -> List[Document]:
    """
    Loads a PDF and splits it into context-aware chunks based on sentence similarity.
    """
    print(f"Starting contextual chunking for: {pdf_path}")
    try:
        pdf_loader = PyMuPDFLoader(pdf_path)  # Previously PyPDFLoader(pdf_path)
        # ----------------------------
        docs = pdf_loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])

        # 1. Split the text into sentences
        sentences = nltk.sent_tokenize(full_text)
        if not sentences:
            print("Warning: No sentences found in document.")
            return []

        print(f"Document split into {len(sentences)} sentences.")

        # 2. Generate embeddings for each sentence
        sentence_embeddings = embedding_model.embed_documents(sentences)
        print("Sentence embeddings generated.")

        # 3. Calculate similarity between adjacent sentences
        similarities = []
        for i in range(len(sentence_embeddings) - 1):
            embedding1 = np.array(sentence_embeddings[i]).reshape(1, -1)
            embedding2 = np.array(sentence_embeddings[i + 1]).reshape(1, -1)
            sim = cosine_similarity(embedding1, embedding2)[0][0]
            similarities.append(sim)

        print("Calculated similarities between adjacent sentences.")

        # 4. Group sentences into chunks based on the similarity threshold
        chunks = []
        current_chunk_sentences = [sentences[0]]

        for i in range(len(similarities)):
            # If similarity is below the threshold, a new topic begins
            if similarities[i] < similarity_threshold:
                # Finalize the current chunk
                chunks.append(" ".join(current_chunk_sentences))
                # Start a new chunk with the next sentence
                current_chunk_sentences = [sentences[i + 1]]
            else:
                # Otherwise, add the next sentence to the current chunk
                current_chunk_sentences.append(sentences[i + 1])

        # Add the last remaining chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        print(f"Grouped sentences into {len(chunks)} contextual chunks.")

        # 5. Create Document objects for each chunk, preserving metadata
        # (This part is similar to your previous logic for metadata)
        path_parts = pdf_path.split(os.sep)
        category = "unknown"
        if "sources" in path_parts:
            category_index = path_parts.index("sources") + 1
            if category_index < len(path_parts):
                category = path_parts[category_index]

        source_filename = os.path.basename(pdf_path)

        chunk_documents = []
        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": source_filename,
                    "category": category,
                    "chunk_number": i + 1
                }
            )
            chunk_documents.append(chunk_doc)

        return chunk_documents

    except Exception as e:
        print(f"Error during contextual chunking for {pdf_path}: {e}")
        raise


def format_metrics_to_question(metrics: Dict[str, Any], additional_info: str = None) -> str:
    # ... (paste your existing format_metrics_to_question function here)
    gender = metrics.get('gender', 'Not provided')
    age = metrics.get('age', 'Not provided')
    bmi = metrics.get('bmi', 'Not provided')
    cholesterol = metrics.get('cholesterol', 'Not provided')
    hdl = metrics.get('hdl', 'Not provided')
    ldl = metrics.get('ldl', 'Not provided')
    triglycerides = metrics.get('triglycerides', 'Not provided')
    fasting_blood_sugar = metrics.get('fasting_blood_sugar', 'Not provided')
    question_parts = [
        f"Gender: {gender}", f"Age: {age}", f"BMI: {bmi}",
        f"Total Cholesterol: {cholesterol} mg/dL", f"HDL: {hdl} mg/dL", f"LDL: {ldl} mg/dL",
        f"Triglycerides: {triglycerides} mg/dL", f"Fasting Blood Sugar: {fasting_blood_sugar} mg/dL"
    ]
    question = ", ".join(part for part in question_parts if "Not provided" not in part)
    if additional_info:
        question += f". Additional information: {additional_info}"
    return question


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

        # --- UPDATE: Use the new wrapper models for list-based tasks ---
        self.overview_chain = self._create_chain(prompts.OVERVIEW_PROMPT_TEMPLATE, HealthOverview)
        self.priorities_chain = self._create_chain(prompts.PRIORITIES_PROMPT_TEMPLATE, TopPriorities)  # Changed
        self.nutrition_chain = self._create_chain(prompts.NUTRITION_PROMPT_TEMPLATE, NutritionPlan)
        self.meal_plan_chain = self._create_chain(prompts.MEAL_PLAN_PROMPT_TEMPLATE, MealPlan)
        self.lifestyle_chain = self._create_chain(prompts.LIFESTYLE_PROMPT_TEMPLATE, LifestylePlan)  # Changed

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

    async def _generate_task(self, chain: Runnable, question: str, documents: str):
        """Helper to run a chain and handle errors gracefully."""
        try:
            return await chain.ainvoke({"question": question, "documents": documents})
        except Exception as e:
            print(f"Error in generating task for chain {chain}: {e}")
            return None

    async def generate_structured_recommendation(
            self, metrics: Dict[str, Any], additional_info: str = None
    ) -> StructuredRecommendationResponse:
        """Orchestrates LLM calls efficiently by chaining context."""
        question = format_metrics_to_question(metrics, additional_info)

        # --- STEP 1: Heavy RAG call for the main overview ---
        print("Step 1: Performing initial RAG call for overview...")
        documents = await self.retriever.ainvoke(question)
        doc_texts = "\n---\n".join([doc.page_content for doc in documents])
        overview_result = await self._generate_task(self.overview_chain, question, doc_texts)

        # If the most important call fails, we can't proceed.
        if not overview_result:
            raise Exception("Failed to generate the core health overview.")

        # --- STEP 2: Create a concise context from the first result ---
        # This new context is much smaller than the original RAG documents.
        concise_context = f"User Profile: {question}\n\nHealth Overview Summary: {overview_result.summary}"
        print("Step 2: Created concise context for subsequent calls.")

        # --- STEP 3: Run remaining tasks concurrently with the smaller context ---
        print("Step 3: Running secondary tasks concurrently...")
        tasks = [
            self._generate_task(self.priorities_chain, concise_context, ""),  # doc_texts is now empty
            self._generate_task(self.nutrition_chain, concise_context, ""),
            self._generate_task(self.meal_plan_chain, concise_context, ""),
            self._generate_task(self.lifestyle_chain, concise_context, ""),
        ]

        secondary_results = await asyncio.gather(*tasks)
        priorities_result, nutrition_result, meal_plan_result, lifestyle_result = secondary_results

        # --- STEP 4: Assemble the final response ---
        final_priorities = priorities_result.priorities if priorities_result else None
        final_lifestyle = lifestyle_result.recommendations if lifestyle_result else None

        return StructuredRecommendationResponse(
            healthOverview=overview_result,
            topPriorities=final_priorities,
            nutritionPlan=nutrition_result,
            mealPlan=meal_plan_result,
            lifestyle=final_lifestyle,
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