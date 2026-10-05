import json
from pathlib import Path
import streamlit as st
from .config import EMBED_MODEL, RAG_TOP_K

ROOT=Path(__file__).resolve().parents[1]
STORE=ROOT/"knowledge_base"/"vector_store"
DEMO=[
{"source":"demo_maintenance_manual.txt","chunk_id":1,"text":"Engine overheating: inspect coolant level, radiator airflow, thermostat and cooling fan. Do not continue operation when temperature remains in the critical range."},
{"source":"demo_maintenance_manual.txt","chunk_id":2,"text":"Low oil pressure: verify oil level and correct grade, inspect leaks, check oil filter and pressure sensor, and inspect the lubrication circuit before returning the vehicle to service."},
{"source":"demo_maintenance_manual.txt","chunk_id":3,"text":"Excessive vibration: inspect engine mounts, rotating assemblies, belts, bearings and drivetrain alignment. Escalate if vibration increases rapidly or is accompanied by abnormal noise."},
{"source":"demo_maintenance_manual.txt","chunk_id":4,"text":"Low battery voltage: check battery terminals, state of charge, charging system and alternator output. Isolate electrical faults before replacing components."},
{"source":"demo_maintenance_manual.txt","chunk_id":5,"text":"Priority guidance: critical conditions require immediate inspection; high priority issues should be inspected within 24 hours; medium issues within 7 days; low-risk conditions can be monitored."}
]

@st.cache_resource(show_spinner=False)
def model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)

def ensure_store():
    STORE.mkdir(parents=True,exist_ok=True)
    idx=STORE/"index.faiss"; meta=STORE/"metadata.json"
    try:
        import faiss
        if idx.exists() and meta.exists():
            try:
                x=faiss.read_index(str(idx))
                m=json.loads(meta.read_text())
                if x.ntotal==len(m): return
            except Exception:
                idx.unlink(missing_ok=True)
        emb=model().encode([x["text"] for x in DEMO],normalize_embeddings=True).astype("float32")
        index=faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
        faiss.write_index(index,str(idx)); meta.write_text(json.dumps(DEMO,indent=2))
    except Exception:
        pass

@st.cache_resource(show_spinner=False)
def retriever():
    ensure_store()
    try:
        import faiss
        index=faiss.read_index(str(STORE/"index.faiss"))
        meta=json.loads((STORE/"metadata.json").read_text())
        m=model()
        def search(query,k=RAG_TOP_K):
            q=m.encode([query],normalize_embeddings=True).astype("float32")
            scores,ids=index.search(q,k)
            return [{**meta[i],"score":float(scores[0][j])} for j,i in enumerate(ids[0]) if i>=0]
        return search
    except Exception:
        def fallback(query,k=RAG_TOP_K):
            terms=set(query.lower().split())
            ranked=sorted(DEMO,key=lambda x:sum(t in x["text"].lower() for t in terms),reverse=True)
            return [{**x,"score":0.0} for x in ranked[:k]]
        return fallback
