# from flask import Flask, render_template, request, jsonify
# import json
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# app = Flask(__name__)

# # ---- Load JSON Knowledge Base ----
# with open("data.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # ---- Safe text extraction ----
# def extract_text(item, key="text"):
#     """
#     Extracts text safely from item.
#     Handles:
#     - Strings
#     - Dictionaries with 'text' key
#     """
#     if isinstance(item, str):
#         return item
#     elif isinstance(item, dict) and key in item and isinstance(item[key], str):
#         return item[key]
#     return ""

# # ---- Process all items ----
# all_texts = [extract_text(item) for item in data if extract_text(item)]
# print(f"Loaded {len(all_texts)} texts.")

# # ---- Load sentence transformer model ----
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # ---- Create embeddings ----
# embeddings = model.encode(all_texts, convert_to_numpy=True)
# dimension = embeddings.shape[1]

# # ---- Setup FAISS index ----
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)
# print(f"FAISS index with {index.ntotal} vectors created.")

# # ---- Flask Routes ----
# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/ask", methods=["POST"])
# def ask():
#     user_query = request.json.get("query")
#     if not user_query:
#         return jsonify({"answer": "Please provide a query."})

#     # Embed query
#     query_vec = model.encode([user_query], convert_to_numpy=True)

#     # Search in FAISS
#     D, I = index.search(query_vec, k=3)  # top 3 results
#     answers = [all_texts[i] for i in I[0]]

#     return jsonify({"answer": " | ".join(answers)})

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

# ---- Load JSON Knowledge Base ----
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ---- Safe text extraction ----
def extract_text(item, key="text"):
    if isinstance(item, str):
        return item
    elif isinstance(item, dict) and key in item and isinstance(item[key], str):
        return item[key]
    return ""

# ---- Process all items ----
all_texts = [extract_text(item) for item in data if extract_text(item)]
print(f"Loaded {len(all_texts)} texts.")

# ---- Load sentence transformer model ----
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Create embeddings ----
embeddings = model.encode(all_texts, convert_to_numpy=True)
dimension = embeddings.shape[1]

# ---- Setup FAISS index ----
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"FAISS index with {index.ntotal} vectors created.")

# ---- Flask Routes ----
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"answers": ["Please provide a query."]})

    # Embed query
    query_vec = model.encode([user_query], convert_to_numpy=True)

    # Search in FAISS
    D, I = index.search(query_vec, k=3)  # top 3 results
    answers = [all_texts[i] for i in I[0]]

    return jsonify({"answers": answers})

if __name__ == "__main__":
    app.run(debug=True)

