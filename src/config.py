import os
import streamlit as st

def secret(name, default=""):
    try:
        v=st.secrets.get(name)
        if v: return str(v)
    except Exception: pass
    return os.getenv(name, default)

GROQ_API_KEY=secret("GROQ_API_KEY")
GROQ_MODEL=secret("GROQ_MODEL","openai/gpt-oss-120b")
EMBED_MODEL=secret("EMBED_MODEL","sentence-transformers/all-MiniLM-L6-v2")
RAG_TOP_K=int(secret("RAG_TOP_K","3"))
