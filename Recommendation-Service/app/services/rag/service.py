# app/services/rag/service.py
import os
from langchain_groq import ChatGroq
from app.core.config import settings
from app.services.vector_store import get_or_create_vector_store, embedding_model
from app.utils.document_processor import _load_and_split_pdf
from app.utils.clinical_interpreter import format_metrics_to_question

from app.services.rag.nodes import RAGNodes
from app.services.rag.graph import build_recommendation_graph


class RAGService:

    def __init__(self):
        self._initialized = None
        if self._initialized: return

        # 1. Init Base Models
        self.llm = ChatGroq(temperature=settings.TEMPERATURE, groq_api_key=settings.GROQ_API_KEY,
                            model_name=settings.LLM_MODEL_NAME)
        self.vectorstore = get_or_create_vector_store()
        self.embedding_model = embedding_model

        # 2. Init Graph Components
        self.nodes = RAGNodes(self.llm, self.vectorstore)
        self.graph = build_recommendation_graph(self.nodes)

        self._initialized = True

    async def generate_structured_recommendation(self, metrics, reported_conditions, reported_habits,
                                                 additional_info=None):
        """API Entry Point for Recommendations"""
        initial_question = format_metrics_to_question(metrics, reported_conditions, reported_habits, additional_info)

        initial_state = {
            "initial_question": initial_question,
            "context_documents": {}, "analyst_summary": None, "overview": None,
            "priorities": None, "nutrition_plan": None, "meal_plan": None,
            "lifestyle": None, "risk_assessment": None, "final_response": None
        }

        final_state = await self.graph.ainvoke(initial_state)
        return final_state["final_response"]

    async def add_pdf_to_index(self, pdf_path: str):
        """API Entry Point for Indexing"""
        doc_splits = _load_and_split_pdf(pdf_path, self.embedding_model, settings.CONTEXTUAL_CHUNK_SIMILARITY_THRESHOLD)
        if doc_splits:
            await self.vectorstore.aadd_documents(doc_splits)

    async def reindex_all_sources(self) -> dict:
        """API Entry Point for Reindexing"""
        # Your existing reindex logic goes here, calling _load_and_split_pdf
        pass