
# from flask import Flask, request, jsonify, render_template
# import json, os, pickle
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from pathlib import Path
# from tqdm import tqdm

# # ------------ CONFIG ------------
# DATA_PATH = "rag_documents.json"
# INDEX_PATH = "faiss.index"
# META_PATH = "metadata.pkl"
# EMB_PATH = "embeddings.npy"

# MODEL_NAME = "all-MiniLM-L6-v2"
# TOP_K = 5
# SIM_THRESHOLD = 0.57    # intent + semantic threshold
# MAX_CHARS = 400

# # ------------ GLOBALS ------------
# app = Flask(__name__)
# _embedder = None
# _index = None
# _metadata = None
# _embeddings = None


# # ------------ LOAD JSON ------------
# def load_docs():
#     if not Path(DATA_PATH).exists():
#         raise FileNotFoundError("rag_documents.json missing")

#     with open(DATA_PATH, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     return data["documents"]


# # ------------ EMBEDDINGS LOADER ------------
# def get_embedder():
#     global _embedder
#     if _embedder is None:
#         _embedder = SentenceTransformer(MODEL_NAME)
#     return _embedder


# # ------------ BUILD / LOAD FAISS INDEX ------------
# def build_index(force=False):
#     global _index, _metadata, _embeddings

#     if Path(INDEX_PATH).exists() and Path(META_PATH).exists() and Path(EMB_PATH).exists() and not force:
#         _index = faiss.read_index(INDEX_PATH)
#         _embeddings = np.load(EMB_PATH)
#         with open(META_PATH, "rb") as f:
#             _metadata = pickle.load(f)
#         return

#     docs = load_docs()
#     _metadata = docs

#     embedder = get_embedder()
#     texts = [d["text"] for d in docs]

#     embs = []
#     B = 50
#     for i in tqdm(range(0, len(texts), B)):
#         batch = texts[i:i + B]
#         arr = embedder.encode(batch, convert_to_numpy=True)
#         embs.append(arr)

#     embs = np.vstack(embs).astype("float32")
#     faiss.normalize_L2(embs)

#     index = faiss.IndexFlatIP(embs.shape[1])
#     index.add(embs)

#     faiss.write_index(index, INDEX_PATH)
#     np.save(EMB_PATH, embs)
#     with open(META_PATH, "wb") as f:
#         pickle.dump(docs, f)

#     _index = index
#     _embeddings = embs


# def ensure_index():
#     if _index is None:
#         build_index(False)


# # ------------ INTENT DETECTION ------------
# def detect_intent(query):
#     q = query.lower()

#     intents = {
#         "award":      ["award", "awards", "prize", "trophy", "won"],
#         "achievement":["achievement", "record", "milestone"],
#         "bio":        ["who is", "biography", "about", "summary"],
#         "birth":      ["born", "birth", "birthplace"],
#         "career":     ["career", "debut", "matches", "team"],
#         "personal":   ["wife", "family", "children"]
#     }

#     for intent, keys in intents.items():
#         for k in keys:
#             if k in q:
#                 return intent

#     return "general"


# # ------------ STRICT ANSWER FILTERING ------------
# def intent_filter(intent, text):
#     t = text.lower()

#     intent_keywords = {
#         "award":      ["award", "prize", "trophy", "won", "winner"],
#         "achievement":["achievement", "record", "milestone"],
#         "bio":        ["born", "biography", "journey", "early life"],
#         "birth":      ["born", "birth", "birthday", "birthplace"],
#         "career":     ["career", "debut", "team", "match"],
#         "personal":   ["family", "wife", "married"]
#     }

#     if intent == "general":
#         return True

#     keys = intent_keywords.get(intent, [])
#     return any(k in t for k in keys)


# # ------------ VECTOR SEARCH WITH INTENT FILTER ------------
# def vector_search(query):
#     ensure_index()
#     embedder = get_embedder()

#     q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
#     faiss.normalize_L2(q_emb)

#     D, I = _index.search(q_emb, TOP_K)

#     intent = detect_intent(query)

#     for idx, score in zip(I[0], D[0]):
#         if idx == -1:
#             continue

#         # not relevant enough → skip
#         if score < SIM_THRESHOLD:
#             continue

#         doc = _metadata[idx]

#         # strict meaning-based filtering
#         if intent_filter(intent, doc["text"]):
#             return doc["text"]

#     return None


# # ------------ CHAT RULES ------------
# def apply_rules(query):
#     q = query.lower().strip()

#     if q in ["bye", "exit", "quit", "goodbye"]:
#         return "Bye! Take care 💙"

#     if len(q) < 2 or not any(c.isalnum() for c in q):
#         return "Invalid question."

#     result = vector_search(query)
#     if not result:
#         return "Sorry, I don't know this. Please ask something from my data."

#     return result[:MAX_CHARS]


# # ------------ ROUTES ------------
# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/query", methods=["POST"])
# def ask():
#     q = (request.json.get("query") or "").strip()
#     if not q:
#         return {"ok": False, "answer": "Invalid question."}

#     ans = apply_rules(q)
#     return {"ok": True, "answer": ans}


# # ------------ RUN ------------
# if __name__ == "__main__":
#     build_index(False)
#     app.run(host="0.0.0.0", port=8080, debug=True)
from flask import Flask, request, jsonify, render_template
import json, os, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

DATA_PATH = "rag_documents.json"
INDEX_PATH = "faiss.index"
META_PATH = "metadata.pkl"
EMB_PATH = "embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
SIM_THRESHOLD = 0.57
MAX_CHARS = 400

app = Flask(__name__)

_embedder = None
_index = None
_metadata = None
_embeddings = None

# ---------------------------------------------------------
# LOADING
# ---------------------------------------------------------

def load_docs():
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError("rag_documents.json missing")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["documents"]


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


# ---------------------------------------------------------
# BUILD / LOAD INDEX
# ---------------------------------------------------------

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

    embedder = get_embedder()
    texts = [d["text"] for d in docs]

    embs = []
    B = 50

    for i in tqdm(range(0, len(texts), B)):
        batch = texts[i:i + B]
        arr = embedder.encode(batch, convert_to_numpy=True)
        embs.append(arr)

    embs = np.vstack(embs).astype("float32")
    faiss.normalize_L2(embs)

    index = faiss.IndexFlatIP(embs.shape[1])
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


# ---------------------------------------------------------
# INTENT SYSTEM (SEMANTIC)
# ---------------------------------------------------------

INTENT_MAP = {
    "award": ["award", "awards", "trophy", "won", "prize"],
    "achievement": ["achievement", "record", "milestone"],
    "bio": ["who is", "biography", "about", "summary"],
    "birth": ["born", "birth", "birthday", "birthplace"],
    "career": ["career", "debut", "team", "match"],
    "personal": ["family", "wife", "children"]
}


def build_intent_embeddings():
    embedder = get_embedder()
    intent_vectors = {}

    for intent, examples in INTENT_MAP.items():
        emb = embedder.encode(examples, convert_to_numpy=True)
        emb = emb.mean(axis=0)
        emb = emb / np.linalg.norm(emb)
        intent_vectors[intent] = emb

    return intent_vectors


INTENT_VECTORS = build_intent_embeddings()


def detect_intent(query):
    embedder = get_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    q_emb = q_emb / np.linalg.norm(q_emb)

    best_intent = "general"
    best_score = 0.0

    for intent, i_emb in INTENT_VECTORS.items():
        score = np.dot(q_emb, i_emb)
        if score > best_score:
            best_score = score
            best_intent = intent

    return best_intent


# ---------------------------------------------------------
# INTENT STRICT FILTER
# ---------------------------------------------------------

INTENT_KEYWORDS = {
    "award": ["award", "trophy", "prize", "won"],
    "achievement": ["achievement", "record"],
    "bio": ["born", "biography", "life"],
    "birth": ["born", "birth", "birthplace"],
    "career": ["career", "team", "debut", "match"],
    "personal": ["family", "wife"]
}

def intent_filter(intent, text):
    if intent == "general":
        return True

    t = text.lower()
    keys = INTENT_KEYWORDS.get(intent, [])
    return any(k in t for k in keys)


# ---------------------------------------------------------
# RAG SEARCH
# ---------------------------------------------------------

def vector_search(query):
    ensure_index()
    embedder = get_embedder()

    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    D, I = _index.search(q_emb, TOP_K)

    intent = detect_intent(query)

    for idx, score in zip(I[0], D[0]):
        if idx == -1 or score < SIM_THRESHOLD:
            continue

        doc = _metadata[idx]

        if intent_filter(intent, doc["text"]):
            return doc["text"]

    return None


# ---------------------------------------------------------
# CHAT RULES
# ---------------------------------------------------------

def apply_rules(query):
    q = query.lower().strip()

    if q in ["bye", "exit", "quit", "goodbye"]:
        return "Bye! Take care 💙"

    if len(q) < 2 or not any(c.isalnum() for c in q):
        return "Invalid question."

    answer = vector_search(query)

    if not answer:
        return "Sorry, I don’t know this. Please ask something from my data."

    return answer[:MAX_CHARS]


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def ask():
    q = (request.json.get("query") or "").strip()
    if not q:
        return {"ok": False, "answer": "Invalid question."}

    ans = apply_rules(q)
    return {"ok": True, "answer": ans}


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    build_index(False)
    app.run(host="0.0.0.0", port=8080, debug=True)
