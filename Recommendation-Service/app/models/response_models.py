from pydantic import BaseModel, Field
from typing import List, Optional

# --- Models for individual LLM call outputs ---

class KeyMetric(BaseModel):
    name: str = Field(description="Name of the health metric, e.g., 'BMI', 'LDL-C'")
    value: float = Field(description="The user's value for the metric")
    unit: str = Field(description="The unit of measurement, e.g., 'kg/m²', 'mg/dL'")
    status: str = Field(description="A qualitative assessment, e.g., 'Normal', 'Good', 'Borderline High'")

class FoodRec(BaseModel):
    group: str = Field(description="The food group or item, e.g., 'Whole-grain red rice'")
    reason: str = Field(description="The reason for the recommendation")
    examples: List[str] = Field(description="Specific food examples")

class HealthOverview(BaseModel):
    summary: str = Field(description="The main paragraph summarizing the user's health.")
    takeaways: List[str] = Field(description="A bulleted list of key takeaways.")
    keyMetrics: List[KeyMetric] = Field(description="A list of key health metrics with their status.")

class NutritionPlan(BaseModel):
    emphasize: List[FoodRec] = Field(description="List of foods to emphasize")
    limit: List[FoodRec] = Field(description="List of foods to limit")

class MealPlan(BaseModel):
    breakfast: str
    lunch: str
    snack: str
    dinner: str

class LifestyleRec(BaseModel):
    topic: str = Field(description="Lifestyle topic, e.g., 'Physical Activity', 'Sleep'")
    recommendation: str
    source: Optional[str] = Field(None, description="The source of the recommendation, if available")

# --- NEW: Wrapper models for list outputs ---
class TopPriorities(BaseModel):
    priorities: List[str] = Field(description="A list of the user's top 3 health priorities.")

class LifestylePlan(BaseModel):
    recommendations: List[LifestyleRec] = Field(description="A list of lifestyle recommendations.")
# ---------------------------------------------


class RiskAssessment(BaseModel):
    risk_level: str = Field(description="A qualitative assessment of the user's overall health risk (e.g., 'Low', 'Moderate', 'High').")
    summary: str = Field(description="A concise paragraph explaining the reasoning for the risk level, summarizing the key factors.")
    next_steps: List[str] = Field(description="A list of 2-3 most critical, actionable next steps for the user.")

# ... (HealthOverview, NutritionPlan, MealPlan, etc. are unchanged)

# --- 2. UPDATE the final response model to include the new section ---
class StructuredRecommendationResponse(BaseModel):
    healthOverview: Optional[HealthOverview] = Field(default_factory=HealthOverview)
    topPriorities: Optional[List[str]] = Field(default_factory=list)
    nutritionPlan: Optional[NutritionPlan] = Field(default_factory=NutritionPlan)
    mealPlan: Optional[MealPlan] = Field(default_factory=MealPlan)
    lifestyle: Optional[List[LifestyleRec]] = Field(default_factory=list)
    riskAssessment: Optional[RiskAssessment] # <-- ADD THIS NEW FIELD