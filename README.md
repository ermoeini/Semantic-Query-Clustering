# Semantic Query Clustering for SEO (Streamlit App)

This Streamlit application allows you to upload a list of search queries and cluster them semantically using OpenAI / OpenRouter embeddings.

It is built for **SEO specialists** who want to:
- Group keywords by user intent
- Identify content hubs, pillar pages, and cluster pages
- Measure semantic distances between offerings
- Decide whether multiple services belong on one domain or require separate branding
- Support **Topical Authority** strategy with real data

---

## 🌍 Repository

git clone git@github.com
:ermoeini/Semantic-Query-Clustering.git


---

## ✨ Features
- Upload `.xlsx` containing a column named: **`Queries`**
- Compute embeddings using:
  - `text-embedding-3-small` (OpenAI native)  
  - or `openai/text-embedding-3-small` via OpenRouter
- **Adjust cluster count (K) anytime** without recomputing embeddings
- Visualizations included:
  - Interactive **UMAP semantic map**
  - **Cluster statistics table**
  - Per-query **similarity lookup**
  - **Heatmap** of similarity matrix
  - **Network graph** of semantic relationships
  - **Hierarchical dendrogram** (optional)
- Export final dataset with cluster assignments
- Helps create **SEO content structure** and **domain strategy**

---

## 🔧 Setup Instructions

### 1) Clone the repository
```bash
git clone git@github.com:ermoeini/Semantic-Query-Clustering.git
cd Semantic-Query-Clustering

2) Install dependencies
pip install -r requirements.txt

3) Add your API key(s)

Create the secrets file:

mkdir -p .streamlit
nano .streamlit/secrets.toml


Put one of the following inside:

For OpenRouter:

OPENROUTER_API_KEY = "your_api_key_here"


For OpenAI:

OPENAI_API_KEY = "your_api_key_here"

4) Run the app
streamlit run streamlit_app.py

📤 Input Format

Your Excel file should contain one column:

Queries
how to buy bitcoin
best crypto wallet
ethereum price prediction
blockchain developer salary

Case-insensitive: Queries / queries / QUERIES are accepted

Blank rows are automatically removed