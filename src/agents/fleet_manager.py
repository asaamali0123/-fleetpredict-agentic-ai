from crewai import Agent
from ..config import GROQ_API_KEY, GROQ_MODEL

def build_agent(llm):
    return Agent(
        role="Fleet Manager / Final Decision Agent",
        goal="Verify the entire chain and produce an executive fleet report.",
        backstory="""You are the Fleet Manager / Final Decision Agent for FleetPredict AI. You work in an industrial fleet setting. Be precise, conservative and explicit about uncertainty.""",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

TASK_INSTRUCTION = """Act as the final verification gate. Reject unsupported recommendations. Explicitly show Evidence -> Diagnosis -> Recommendation and produce an executive report."""
