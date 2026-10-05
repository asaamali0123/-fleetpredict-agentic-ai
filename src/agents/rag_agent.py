from crewai import Agent
from ..config import GROQ_API_KEY, GROQ_MODEL

def build_agent(llm):
    return Agent(
        role="Maintenance Knowledge / RAG Agent",
        goal="Ground maintenance reasoning in retrieved maintenance evidence.",
        backstory="""You are the Maintenance Knowledge / RAG Agent for FleetPredict AI. You work in an industrial fleet setting. Be precise, conservative and explicit about uncertainty.""",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

TASK_INSTRUCTION = """Use ONLY the supplied retrieved chunks as factual maintenance evidence. Return relevant evidence, component, procedure and source. Do not invent manual content."""
