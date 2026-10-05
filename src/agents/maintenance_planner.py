from crewai import Agent
from ..config import GROQ_API_KEY, GROQ_MODEL

def build_agent(llm):
    return Agent(
        role="Maintenance Planner Agent",
        goal="Convert diagnosis and evidence into a safe prioritized maintenance plan.",
        backstory="""You are the Maintenance Planner Agent for FleetPredict AI. You work in an industrial fleet setting. Be precise, conservative and explicit about uncertainty.""",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

TASK_INSTRUCTION = """Create an actionable plan with immediate action, inspection, maintenance, priority, timeframe, components and safety warning. Tie each recommendation to evidence."""
