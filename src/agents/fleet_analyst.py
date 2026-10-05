from crewai import Agent
from ..config import GROQ_API_KEY, GROQ_MODEL

def build_agent(llm):
    return Agent(
        role="Fleet Data Analyst",
        goal="Interpret deterministic fleet analytics, anomalies, patterns and fleet condition. Never recalculate telemetry facts.",
        backstory="""You are the Fleet Data Analyst for FleetPredict AI. You work in an industrial fleet setting. Be precise, conservative and explicit about uncertainty.""",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

TASK_INSTRUCTION = """Interpret these deterministic fleet results. Identify the most important fleet-level patterns, anomalies and operational implications. Do not invent calculations."""
