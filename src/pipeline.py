import json
from .config import GROQ_API_KEY, EMBED_MODEL, RAG_TOP_K
from .rag import retriever
from .crew import build_crew

def run_diagnosis(analysis, vehicle_id):
    if not GROQ_API_KEY:
        return {"available":False,"error":"GROQ_API_KEY is missing. Add it to Streamlit Secrets."}
    try:
        fleet=analysis["vehicle_summary"].to_dict("records")
        selected=next(x for x in fleet if str(x["vehicle_id"])==str(vehicle_id))
        query=f"maintenance procedure for {selected['main_issue']}"
        evidence=retriever()(query,RAG_TOP_K)
        fleet_text=json.dumps(fleet,indent=2)
        vehicle_text=json.dumps(selected,indent=2)
        rag_text=json.dumps(evidence,indent=2)
        crew,tasks=build_crew(fleet_text,vehicle_text,rag_text)
        crew.kickoff()
        outs=[str(t.output.raw) for t in tasks]
        return {"available":True,"fleet_analyst":outs[0],"health_agent":outs[1],
                "root_cause":outs[2],"rag_agent":outs[3],"maintenance_planner":outs[4],
                "maintenance_plan":outs[4],"fleet_manager":outs[5],"rag_evidence":evidence}
    except Exception as e:
        return {"available":False,"error":f"{type(e).__name__}: {e}"}
