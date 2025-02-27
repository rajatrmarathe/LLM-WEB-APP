from flask import Flask, request, jsonify, render_template
import faiss
#import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
import torch
import os
from dotenv import load_dotenv


# Initialize Flask app
app = Flask(__name__)

load_dotenv()

#openai.api_key = os.getenv("OPENAI_API_KEY")  # Fetch API key from environment variable
#print(openai.api_key)
#print(os.environ)

# Retrieve the API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the key is loaded correctly
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.") 

# Load the embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device=device)

# Load sentences from Excel file
df = pd.read_excel("pad_sentences_for_search_engine.xlsx", engine="openpyxl")  
sentences = df["sentence_text"].dropna().tolist()  

# Generate embeddings
sentence_vectors = model.encode(sentences, convert_to_numpy=True, batch_size=256, show_progress_bar=True)

# Create FAISS index
dimension = sentence_vectors.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(sentence_vectors)  


def search(query, top_k=5, mode="both"):
    """Search function with vector and literal matching."""
    results = []

    if mode in ["semantic", "both"]:
        query_vector = model.encode([query], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_vector, top_k)
        results.extend(
            [{"sentence": sentences[i], "score": float(distances[0][j]), "method": "semantic"} for j, i in enumerate(indices[0])]
        )

    if mode in ["literal", "both"]:
        literal_matches = [
            {"sentence": sent, "score": 0.0, "method": "literal"}
            for sent in sentences if query.lower() in sent.lower()
        ]
        results.extend(literal_matches[:top_k])

    return sorted(results, key=lambda x: x["score"])[:top_k]

#query = "project management"
#results = search(query)
#for r in results:
 #   print(f"Sentence: {r['sentence']} | Score: {r['score']}")

def summarize_sentences(sentences):
    """Summarize top sentences using OpenAI API."""
    try:
        prompt = "Summarize the following sentences:\n\n" + "\n".join(sentences)
        
        response = openai.chat.completions.create(  # Updated syntax
            model="gpt-4o-mini",  # Used gpt-4o-mini
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error summarizing: {str(e)}"


@app.route("/", methods=["GET", "POST"])
def home():
    """Render home page with search functionality."""
    results, summary = [], ""
    query = ""
    search_mode = "both"  # Default to both if nothing is selected

    if request.method == "POST":
        query = request.form["query"]
        search_mode = request.form.get("search_mode", "both")  # Default to both if no mode selected
        
        if query:
            results = search(query, top_k=5, mode=search_mode)
            # Optionally, summarize results
            if results:
                summary = summarize_sentences([r["sentence"] for r in results])

    return render_template("index.html", query=query, results=results, summary=summary)


if __name__ == "__main__":
    app.run(debug=True)
