
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import analyze_fleet, vehicle_diagnostics, what_if
from src.config import GROQ_API_KEY, GROQ_MODEL
from src.pipeline import run_diagnosis

st.set_page_config(page_title="FleetPredict AI", page_icon="🚛", layout="wide")

st.markdown("""
<style>
.block-container {max-width:1450px;padding-top:1rem}
.kpi{padding:16px 18px;border:1px solid #29364f;border-radius:14px;background:#111a2b}
.kpi .v{font-size:30px;font-weight:800}.kpi .l{font-size:12px;opacity:.7}
</style>
""", unsafe_allow_html=True)

def kpi(label, value):
    return f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>'

st.title("🚛 FleetPredict AI")
st.caption("Agentic Fleet Diagnostic & Predictive Maintenance Copilot")

with st.sidebar:
    st.header("Control Center")
    uploaded = st.file_uploader("Upload telemetry CSV", type=["csv"])
    use_demo = st.checkbox("Use demo telemetry", value=uploaded is None)
    if uploaded is not None:
        df = pd.read_csv(uploaded); source = uploaded.name
    elif use_demo:
        df = pd.read_csv("data/sample_telemetry.csv"); source = "sample_telemetry.csv"
    else:
        st.info("Upload a CSV or enable demo telemetry."); st.stop()

analysis = analyze_fleet(df)
ranking = analysis["vehicle_summary"]
vehicles = ranking["vehicle_id"].astype(str).tolist()

with st.sidebar:
    vehicle = st.selectbox("Select vehicle", vehicles)
    run = st.button("🧠 Run Agentic Diagnosis", use_container_width=True)
    report = st.button("📋 Generate AI Fleet Report", use_container_width=True)
    st.divider()
    st.caption(f"Data: `{source}`")
    st.caption(f"Groq model: `{GROQ_MODEL}`")
    if not GROQ_API_KEY:
        st.warning("GROQ_API_KEY is not configured.")

st.subheader("Fleet Command Dashboard")
a,b,c,d=st.columns(4)
a.markdown(kpi("Total Vehicles", analysis["kpis"]["total"]), unsafe_allow_html=True)
b.markdown(kpi("Healthy", analysis["kpis"]["healthy"]), unsafe_allow_html=True)
c.markdown(kpi("Watch / Warning", analysis["kpis"]["warning_watch"]), unsafe_allow_html=True)
d.markdown(kpi("Critical", analysis["kpis"]["critical"]), unsafe_allow_html=True)

l,r=st.columns([1.25,1])
with l:
    st.markdown("### Fleet Risk Ranking")
    st.dataframe(ranking[["vehicle_id","health_score","risk_level","risk_score","main_issue","anomaly_count"]]
        .rename(columns={"vehicle_id":"Vehicle","health_score":"Health","risk_level":"Risk",
                         "risk_score":"Risk Score","main_issue":"Main Issue","anomaly_count":"Anomalies"}),
        use_container_width=True, hide_index=True)
with r:
    fig=px.bar(ranking,x="vehicle_id",y="risk_score",color="risk_level",
               title="Risk Score by Vehicle")
    fig.update_layout(height=330,margin=dict(l=10,r=10,t=45,b=10))
    st.plotly_chart(fig,use_container_width=True)

st.markdown("### Anomaly Overview")
x,y=st.columns(2)
with x:
    st.dataframe(analysis["anomaly_counts"].rename(columns={"metric":"Sensor","count":"Anomalies"}),
                 use_container_width=True,hide_index=True)
with y:
    st.info("**Separation of concerns:** Pandas calculates facts; CrewAI interprets them; RAG supplies maintenance evidence.")

diag=vehicle_diagnostics(df,vehicle)
st.markdown(f"## 🔎 Vehicle Diagnostic — {vehicle}")
x,y,z=st.columns(3)
x.metric("Health Score",diag["health_score"])
y.metric("Risk",diag["risk_level"])
z.metric("Risk Score",diag["risk_score"])
st.dataframe(pd.DataFrame([diag["sensors"]]),use_container_width=True,hide_index=True)
if diag["anomalies"]: st.warning(" • ".join(diag["anomalies"]))
else: st.success("No threshold anomalies detected.")

st.markdown("### Evidence → Diagnosis → Recommendation")
x,y,z=st.columns(3)
x.markdown("**Evidence**"); x.write(diag["evidence_summary"])
y.markdown("**Diagnosis**"); y.write(diag["diagnosis_hint"])
z.markdown("**Recommendation**"); z.write(diag["priority_action"])

if run or report:
    with st.spinner("CrewAI is orchestrating six specialist agents…"):
        result=run_diagnosis(analysis,vehicle)
    st.session_state["result"]=result

if "result" in st.session_state:
    result=st.session_state["result"]
    if not result["available"]:
        st.warning("AI agent analysis unavailable — deterministic diagnostics are still available.")
        st.caption(result.get("error","Unknown AI error"))
    else:
        st.markdown("## 🤖 Multi-Agent Analysis")
        tabs=st.tabs(["1 Fleet Analyst","2 Health Agent","3 Root Cause","4 RAG Agent","5 Maintenance Planner","6 Fleet Manager"])
        for tab,key in zip(tabs,["fleet_analyst","health_agent","root_cause","rag_agent","maintenance_planner","fleet_manager"]):
            with tab: st.markdown(result[key])
        st.markdown("### 📚 Retrieved Maintenance Evidence")
        for e in result["rag_evidence"]:
            with st.expander(f"{e['source']} • chunk {e['chunk_id']}"):
                st.write(e["text"])
        st.markdown("### 🛠 Maintenance Plan")
        st.write(result["maintenance_plan"])

st.markdown("## 🧪 What-If Simulator")
s=diag["sensors"]
q1,q2,q3,q4=st.columns(4)
et=q1.number_input("Engine temp °C",value=float(s.get("engine_temp") or 95),step=1.0)
op=q2.number_input("Oil pressure",value=float(s.get("oil_pressure") or 45),step=1.0)
vi=q3.number_input("Vibration",value=float(s.get("vibration") or 3),step=.5)
bv=q4.number_input("Battery voltage",value=float(s.get("battery_voltage") or 12.4),step=.1)
w=what_if(et,op,vi,bv)
q1.metric("New Health",w["health_score"]); q2.metric("New Risk",w["risk_level"]); q3.metric("Anomalies",len(w["anomalies"]))
if w["anomalies"]: st.warning(" • ".join(w["anomalies"]))
else: st.success("No simulated threshold anomalies.")

st.divider()
st.caption("Advisory system. Maintenance decisions require qualified personnel and applicable OEM procedures.")
