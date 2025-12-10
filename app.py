# from flask import Flask, request, jsonify, render_template
# import json
# import os
# import numpy as np
# import faiss
# import pickle
# from sentence_transformers import SentenceTransformer
# from pathlib import Path
# from tqdm import tqdm

# # -------------- CONFIG ----------------

# DATA_PATHS = [
#     "rag_documents.json"
# ]

# INDEX_PATH = "./faiss.index"
# META_PATH = "./metadata.pkl"
# EMB_PATH = "./embeddings.npy"

# MODEL_NAME = "all-MiniLM-L6-v2"
# EMB_DIM = 384
# TOP_K = 5

# # ---------------------------------------

# app = Flask(__name__)

# _embedder = None
# _index = None
# _metadata = None
# _embeddings = None


# # ------------ HELPERS --------------------

# def find_json():
#     for p in DATA_PATHS:
#         if Path(p).exists():
#             return p
#     raise FileNotFoundError("rag_documents.json not found.")


# def load_docs():
#     path = find_json()
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     docs = data["documents"]
#     return docs


# def get_embedder():
#     global _embedder
#     if _embedder is None:
#         print("Loading embedding model:", MODEL_NAME)
#         _embedder = SentenceTransformer(MODEL_NAME)
#     return _embedder


# def build_index(force=False):
#     global _index, _metadata, _embeddings

#     if (
#         Path(INDEX_PATH).exists()
#         and Path(META_PATH).exists()
#         and Path(EMB_PATH).exists()
#         and not force
#     ):
#         print("Loading saved FAISS index & metadata...")
#         _index = faiss.read_index(INDEX_PATH)
#         _embeddings = np.load(EMB_PATH)
#         with open(META_PATH, "rb") as f:
#             _metadata = pickle.load(f)
#         return

#     docs = load_docs()
#     _metadata = docs

#     texts = [d["text"] for d in docs]

#     embedder = get_embedder()

#     print("Embedding documents...")
#     B = 64
#     embs = []
#     for i in tqdm(range(0, len(texts), B)):
#         batch = texts[i: i + B]
#         arr = embedder.encode(batch, convert_to_numpy=True)
#         embs.append(arr)

#     embs = np.vstack(embs).astype("float32")

#     faiss.normalize_L2(embs)

#     index = faiss.IndexFlatIP(EMB_DIM)
#     index.add(embs)

#     # Save
#     faiss.write_index(index, INDEX_PATH)
#     np.save(EMB_PATH, embs)
#     with open(META_PATH, "wb") as f:
#         pickle.dump(docs, f)

#     _index = index
#     _embeddings = embs

#     print("Index built successfully.")


# def ensure_index():
#     if _index is None:
#         build_index(force=False)


# def vector_search(query, top_k):
#     ensure_index()

#     embedder = get_embedder()
#     q = embedder.encode([query], convert_to_numpy=True).astype("float32")
#     faiss.normalize_L2(q)

#     D, I = _index.search(q, top_k)

#     results = []
#     for idx, score in zip(I[0], D[0]):
#         if idx == -1:
#             continue
#         doc = _metadata[idx]
#         results.append({
#             "id": doc["id"],
#             "title": doc["title"],
#             "text": doc["text"],
#             "score": float(score)
#         })
#     return results


# def build_answer(query, retrieved):
#     if not retrieved:
#         return "No answer found."

#     summary = ""
#     for r in retrieved:
#         summary += f"\n\n---\nSOURCE: {r['title']}\n{r['text']}\n"

#     return summary.strip()


# # ----------- ROUTES ----------------------

# @app.route("/")
# def index_page():
#     return render_template("index.html")


# @app.route("/query", methods=["POST"])
# def http_query():
#     data = request.json
#     query = data.get("query")
#     top_k = int(data.get("top_k", TOP_K))

#     if not query:
#         return {"ok": False, "error": "Query required"}, 400

#     retrieved = vector_search(query, top_k)
#     answer = build_answer(query, retrieved)

#     return {
#         "ok": True,
#         "answer": answer
#     }


# if __name__ == "__main__":
#     build_index(force=False)
#     app.run(debug=True)
from flask import Flask, request, jsonify, render_template
import json
import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ---------------------

DATA_PATHS = ["rag_documents.json"]
INDEX_PATH = "./faiss.index"
META_PATH = "./metadata.pkl"
EMB_PATH = "./embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"
EMB_DIM = 384
TOP_K = 5

# ---------------------------------------------

app = Flask(__name__)

_embedder = None
_index = None
_metadata = None
_embeddings = None


# ---------------- HELPERS ---------------------

def find_json():
    for p in DATA_PATHS:
        if Path(p).exists():
            return p
    raise FileNotFoundError("rag_documents.json not found.")


def load_docs():
    path = find_json()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["documents"]


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def build_index(force=False):
    global _index, _metadata, _embeddings

    if Path(INDEX_PATH).exists() and Path(META_PATH).exists() and Path(EMB_PATH).exists() and not force:
        _index = faiss.read_index(INDEX_PATH)
        _embeddings = np.load(EMB_PATH)
        with open(META_PATH, "rb") as f:
            _metadata = pickle.load(f)
        return

    docs = load_docs()
    _metadata = docs

    texts = [d["text"] for d in docs]
    embedder = get_embedder()

    embs = []
    B = 64
    for i in tqdm(range(0, len(texts), B)):
        batch = texts[i:i+B]
        arr = embedder.encode(batch, convert_to_numpy=True)
        embs.append(arr)

    embs = np.vstack(embs).astype("float32")
    faiss.normalize_L2(embs)

    index = faiss.IndexFlatIP(EMB_DIM)
    index.add(embs)

    faiss.write_index(index, INDEX_PATH)
    np.save(EMB_PATH, embs)
    with open(META_PATH, "wb") as f:
        pickle.dump(docs, f)

    _index = index
    _embeddings = embs


def ensure_index():
    if _index is None:
        build_index(False)


def vector_search(query, top_k):
    ensure_index()
    embedder = get_embedder()

    q = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)

    D, I = _index.search(q, top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        doc = _metadata[idx]
        results.append(doc)
    return results


# ---------------- CHATBOT RULES ---------------------

def apply_chat_rules(query, retrieved):
    q = query.lower().strip()

    # 1. Exit/bye handling
    bye_words = ["bye", "exit", "quit", "goodbye"]
    if q in bye_words:
        return "Bye! Take care ❤"

    # 2. Invalid question
    if len(query.strip()) < 2 or not any(c.isalnum() for c in query):
        return "Invalid question."

    # 3. No RAG result
    if not retrieved:
        return "No answer found."

    # 4. Short exact answer (no extra words)
    return retrieved[0]["text"][:500]   # limit 500 chars (safe)


# ---------------- ROUTES ---------------------

@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query_api():
    data = request.json
    query = data.get("query")

    if not query:
        return {"ok": False, "answer": "Invalid question."}

    retrieved = vector_search(query, TOP_K)
    final_answer = apply_chat_rules(query, retrieved)

    return {"ok": True, "answer": final_answer}


# ---------------- RUN ------------------------

if __name__ == "__main__":
    build_index(False)
    app.run(host="0.0.0.0", port=8080, debug=True)