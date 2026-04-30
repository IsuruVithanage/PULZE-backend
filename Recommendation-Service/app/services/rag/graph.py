# app/services/rag/graph.py
from langgraph.graph import StateGraph, START, END
from app.services.rag.state import RecommendationState
from app.services.rag.nodes import RAGNodes

def build_recommendation_graph(nodes: RAGNodes):
    workflow = StateGraph(RecommendationState)

    # Add Nodes
    workflow.add_node("librarian", nodes.retrieve_context)
    workflow.add_node("analyst", nodes.analyst_overview)
    workflow.add_node("nutritionist", nodes.nutrition_specialist)
    workflow.add_node("meal_planner", nodes.meal_plan_specialist)
    workflow.add_node("lifestyle_coach", nodes.lifestyle_specialist)
    workflow.add_node("risk_assessor", nodes.risk_specialist)
    workflow.add_node("health_planner", nodes.priorities_specialist)
    workflow.add_node("synthesizer", nodes.synthesizer)

    workflow.add_edge(START, "librarian")
    workflow.add_edge("librarian", "analyst")

    specialists = ["nutritionist", "meal_planner", "lifestyle_coach", "risk_assessor", "health_planner"]

    # Fan-OUT: analyst broadcasts to all specialists in parallel
    for specialist in specialists:
        workflow.add_edge("analyst", specialist)

    # Fan-IN: wait for ALL specialists to finish before synthesizer runs (pass a list)
    workflow.add_edge(specialists, "synthesizer")

    workflow.add_edge("synthesizer", END)

    return workflow.compile()