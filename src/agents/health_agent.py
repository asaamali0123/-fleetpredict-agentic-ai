from crewai import Agent
from ..config import GROQ_API_KEY, GROQ_MODEL

def build_agent(llm):
    return Agent(
        role="Vehicle Health & Risk Agent",
        goal="Interpret deterministic health scores, risk levels and highest-risk vehicles.",
        backstory="""You are the Vehicle Health & Risk Agent for FleetPredict AI. You work in an industrial fleet setting. Be precise, conservative and explicit about uncertainty.""",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

TASK_INSTRUCTION = """Evaluate the supplied deterministic health/risk results. Explain severity and why the selected vehicle is risky. Do not change the scores."""
