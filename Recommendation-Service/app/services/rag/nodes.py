# app/services/rag/nodes.py
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from app.models.response_models import *
from app.prompts import prompts
from app.services.rag.state import RecommendationState
import asyncio


class RAGNodes:
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore

        # Initialize chains
        self.overview_chain = self._create_chain(prompts.OVERVIEW_PROMPT_TEMPLATE, HealthOverview)
        self.priorities_chain = self._create_chain(prompts.PRIORITIES_PROMPT_TEMPLATE, TopPriorities)
        self.nutrition_chain = self._create_chain(prompts.NUTRITION_PROMPT_TEMPLATE, NutritionPlan)
        self.meal_plan_chain = self._create_chain(prompts.MEAL_PLAN_PROMPT_TEMPLATE, MealPlan)
        self.lifestyle_chain = self._create_chain(prompts.LIFESTYLE_PROMPT_TEMPLATE, LifestylePlan)
        self.risk_assessment_chain = self._create_chain(prompts.RISK_ASSESSMENT_PROMPT_TEMPLATE, RiskAssessment)

    def _create_chain(self, prompt_template: str, pydantic_object: Any):
        parser = PydanticOutputParser(pydantic_object=pydantic_object)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "documents"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return prompt | self.llm | parser  # ← use parser directly

    # --- THE AGENTS (NODES) ---

    async def retrieve_context(self, state: RecommendationState):
        """Librarian Agent: Fetches all required context."""
        question = state["initial_question"]

        async def fetch(category):
            retriever = self.vectorstore.as_retriever(
                search_kwargs={'k': 4, 'filter': {"category": category}}
            )
            docs = await retriever.ainvoke(question)
            return "\n---\n".join([doc.page_content for doc in docs])

        # Run sequentially — Pinecone's async session can't handle concurrent calls
        general = await fetch("general")
        dietary = await fetch("dietary")
        risk = await fetch("risk_assessment")

        return {"context_documents": {"general": general, "dietary": dietary, "risk": risk}}

    async def analyst_overview(self, state: RecommendationState):
        """Chief Analyst Agent"""
        result = await self.overview_chain.ainvoke({
            "question": state["initial_question"],
            "documents": state["context_documents"].get("general", "")
        })
        return {"overview": result, "analyst_summary": result.summary}

    async def nutrition_specialist(self, state: RecommendationState):
        """Nutrition Agent"""
        result = await self.nutrition_chain.ainvoke({
            "question": state["analyst_summary"],
            "documents": state["context_documents"].get("dietary", "")
        })
        return {"nutrition_plan": result}

    async def meal_plan_specialist(self, state: RecommendationState):
        """Meal Planner Agent"""
        result = await self.meal_plan_chain.ainvoke({
            "question": state["analyst_summary"],
            "documents": state["context_documents"].get("dietary", "")
        })
        return {"meal_plan": result}

    async def lifestyle_specialist(self, state: RecommendationState):
        """Lifestyle Agent"""
        result = await self.lifestyle_chain.ainvoke({
            "question": state["analyst_summary"],
            "documents": state["context_documents"].get("general", "")
        })
        return {"lifestyle": result}

    async def risk_specialist(self, state: RecommendationState):
        """Risk Assessor Agent"""
        result = await self.risk_assessment_chain.ainvoke({
            "question": state["analyst_summary"],
            "documents": state["context_documents"].get("risk", "")
        })
        return {"risk_assessment": result}

    async def priorities_specialist(self, state: RecommendationState):
        """Health Planner Agent"""
        result = await self.priorities_chain.ainvoke({
            "question": state["analyst_summary"],
            "documents": state["context_documents"].get("general", "")
        })
        return {"priorities": result}

    def synthesizer(self, state: RecommendationState):
        """Synthesizer Agent: Compiles final JSON"""
        priorities = state["priorities"].priorities if state["priorities"] else None
        lifestyle = state["lifestyle"].recommendations if state["lifestyle"] else None

        response = StructuredRecommendationResponse(
            healthOverview=state["overview"],
            topPriorities=priorities,
            nutritionPlan=state["nutrition_plan"],
            mealPlan=state["meal_plan"],
            lifestyle=lifestyle,
            riskAssessment=state["risk_assessment"],
        )
        return {"final_response": response}