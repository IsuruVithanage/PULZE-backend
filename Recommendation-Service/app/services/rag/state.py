# app/services/rag/state.py
from typing import TypedDict, Optional, Dict, Any
from app.models.response_models import StructuredRecommendationResponse


class RecommendationState(TypedDict):
    initial_question: str
    context_documents: Dict[str, str]
    analyst_summary: Optional[str]

    # Agent Outputs
    overview: Optional[Any]
    priorities: Optional[Any]
    nutrition_plan: Optional[Any]
    meal_plan: Optional[Any]
    lifestyle: Optional[Any]
    risk_assessment: Optional[Any]

    # Final Synthesized Output
    final_response: Optional[StructuredRecommendationResponse]