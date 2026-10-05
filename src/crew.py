from crewai import Agent, Crew, LLM, Process, Task
from .config import GROQ_API_KEY, GROQ_MODEL
from .agents.fleet_analyst import build_agent as build_fleet, TASK_INSTRUCTION as FLEET_TASK
from .agents.health_agent import build_agent as build_health, TASK_INSTRUCTION as HEALTH_TASK
from .agents.root_cause_agent import build_agent as build_root, TASK_INSTRUCTION as ROOT_TASK
from .agents.rag_agent import build_agent as build_rag, TASK_INSTRUCTION as RAG_TASK
from .agents.maintenance_planner import build_agent as build_plan, TASK_INSTRUCTION as PLAN_TASK
from .agents.fleet_manager import build_agent as build_manager, TASK_INSTRUCTION as MANAGER_TASK

def groq_llm():
    # Current CrewAI-compatible OpenAI endpoint configuration for Groq.
    return LLM(
        model=GROQ_MODEL,
        custom_openai=True,
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=1400
    )

def build_crew(fleet_context, vehicle_context, rag_context):
    llm=groq_llm()
    a1=build_fleet(llm); a2=build_health(llm); a3=build_root(llm)
    a4=build_rag(llm); a5=build_plan(llm); a6=build_manager(llm)

    t1=Task(description=FLEET_TASK+"\n\nFLEET DATA:\n"+fleet_context,expected_output="Concise fleet interpretation.",agent=a1)
    t2=Task(description=HEALTH_TASK+"\n\nVEHICLE DATA:\n"+vehicle_context,expected_output="Risk interpretation.",agent=a2,context=[t1])
    t3=Task(description=ROOT_TASK+"\n\nVEHICLE DATA:\n"+vehicle_context,expected_output="Root-cause hypotheses with confidence.",agent=a3,context=[t1,t2])
    t4=Task(description=RAG_TASK+"\n\nRETRIEVED EVIDENCE:\n"+rag_context,expected_output="Evidence-grounded maintenance interpretation.",agent=a4,context=[t3])
    t5=Task(description=PLAN_TASK+"\n\nVEHICLE DATA:\n"+vehicle_context+"\nRAG EVIDENCE:\n"+rag_context,
             expected_output="Prioritized maintenance plan.",agent=a5,context=[t3,t4])
    t6=Task(description=MANAGER_TASK+"\n\nFLEET:\n"+fleet_context+"\nVEHICLE:\n"+vehicle_context+
             "\nRAG:\n"+rag_context+
             "\nProduce: Fleet Overview; Highest Risk Vehicles; Major Issues; Root Cause Summary; Evidence; Recommended Actions; Priority; Fleet Recommendations; Evidence -> Diagnosis -> Recommendation.",
             expected_output="Executive verified fleet report.",agent=a6,context=[t1,t2,t3,t4,t5])
    crew=Crew(agents=[a1,a2,a3,a4,a5,a6],tasks=[t1,t2,t3,t4,t5,t6],
              process=Process.sequential,verbose=False)
    return crew,(t1,t2,t3,t4,t5,t6)
