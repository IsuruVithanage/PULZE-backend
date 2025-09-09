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


# --- The Final, Assembled Response Model (Unchanged) ---
class StructuredRecommendationResponse(BaseModel):
    healthOverview: Optional[HealthOverview]
    topPriorities: Optional[List[str]]
    nutritionPlan: Optional[NutritionPlan]
    mealPlan: Optional[MealPlan]
    lifestyle: Optional[List[LifestyleRec]]