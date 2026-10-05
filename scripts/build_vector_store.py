from pathlib import Path
import json
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

ROOT=Path(__file__).resolve().parents[1]
manuals=ROOT/"knowledge_base"/"manuals"; store=ROOT/"knowledge_base"/"vector_store"
model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
items=[]
for pdf in manuals.glob("*.pdf"):
    text="\n".join((p.extract_text() or "") for p in PdfReader(str(pdf)).pages)
    text=" ".join(text.split())
    size,overlap=900,120
    start=0; i=1
    while start<len(text):
        end=min(len(text),start+size)
        items.append({"source":pdf.name,"chunk_id":i,"text":text[start:end]})
        if end==len(text): break
        start=end-overlap; i+=1
if not items:
    raise SystemExit("No PDF found. Put a PDF in knowledge_base/manuals/ and rerun.")
emb=model.encode([x["text"] for x in items],normalize_embeddings=True).astype("float32")
index=faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
store.mkdir(parents=True,exist_ok=True)
faiss.write_index(index,str(store/"index.faiss"))
(store/"metadata.json").write_text(json.dumps(items,ensure_ascii=False,indent=2))
print(f"Built {len(items)} chunks.")
