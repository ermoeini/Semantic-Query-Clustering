# streamlit_app.py
import os
import io
import math
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import umap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from pyvis.network import Network
import tempfile
from openai import OpenAI
import pickle

# ---------------------------
# Config / helpers
# ---------------------------
st.set_page_config(layout="wide", page_title="Semantic Query Clustering (SEO)")

# Try environment variables first, then Streamlit secrets
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY") if "OPENROUTER_API_KEY" in st.secrets else None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None

# Determine which backend to use
USE_OPENROUTER = bool(OPENROUTER_API_KEY) and not OPENAI_API_KEY
if USE_OPENROUTER:
    API_KEY = OPENROUTER_API_KEY
else:
    API_KEY = OPENAI_API_KEY

if not API_KEY:
    st.warning("Set OPENROUTER_API_KEY or OPENAI_API_KEY as an env var or in Streamlit secrets (`.streamlit/secrets.toml`). Embeddings won't run without a key.")

# Choose embedding model name depending on backend
# OpenRouter often expects the OpenAI-prefixed model name; OpenAI uses the plain model id.
EMBED_MODEL = "openai/text-embedding-3-small" if USE_OPENROUTER else "text-embedding-3-small"

# Session-state holder
if "state" not in st.session_state:
    st.session_state.state = {
        "df": None,
        "texts": None,
        "embeddings": None,
        "umap_2d": None,
        "similarity": None,
        "clusters": None
    }

def read_xlsx(contents) -> pd.DataFrame:
    x = pd.read_excel(io.BytesIO(contents))
    cols = [c for c in x.columns if c.lower() == "queries"]
    if not cols:
        raise ValueError("No column named 'Queries' found (case-insensitive).")
    df = x[[cols[0]]].dropna().rename(columns={cols[0]: "Queries"})
    df["Queries"] = df["Queries"].astype(str).str.strip()
    df = df[df["Queries"] != ""].reset_index(drop=True)
    return df

def get_openai_client():
    """
    Return an OpenAI client configured either for OpenRouter or for OpenAI.
    """
    if not API_KEY:
        raise RuntimeError("No API key configured.")
    if USE_OPENROUTER:
        # OpenRouter uses a different base_url
        client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")
    else:
        client = OpenAI(api_key=API_KEY)
    return client

@st.cache_data(show_spinner=False)
def compute_embeddings_openai(texts: list):
    """
    Compute embeddings using openai.OpenAI client (v1+).
    This function is cached by Streamlit so re-runs with the same inputs will reuse results.
    """
    client = get_openai_client()
    batch_size = 128
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # call the client embeddings endpoint
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        # resp.data is a list of objects with .embedding attribute
        for item in resp.data:
            # item.embedding should be a list[float]
            vectors.append(item.embedding)
    return np.array(vectors, dtype=np.float32)

def compute_umap(embeddings, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(embeddings)

def make_similarity_matrix(embeddings):
    return cosine_similarity(embeddings)

def cluster_kmeans(embeddings, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km

def hierarchy_linkage(embeddings, method="ward"):
    return linkage(embeddings, method=method)

# ---------------------------
# UI
# ---------------------------
st.title("Semantic Query Clustering — SEO strategy helper")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 1) Upload queries (.xlsx)")
    uploaded = st.file_uploader("Upload an Excel file with a column named `Queries`", type=["xlsx"])
    st.markdown("### 2) Embedding options")
    embed_button = st.button("Compute embeddings / (re)compute")
    st.write("Model:", EMBED_MODEL)
    st.info("Embeddings are cached for the session. Changing cluster count won't force re-embedding.")

with col2:
    st.markdown("### Controls")
    k = st.slider("Number of clusters (K)", min_value=2, max_value=25, value=5, step=1)
    st.checkbox("Show dendrogram", key="show_dend")
    st.checkbox("Show network graph", key="show_net")
    st.checkbox("Show similarity matrix heatmap", key="show_sim")
    st.number_input("UMAP neighbors", key="umap_neighbors", value=15, min_value=5, max_value=200)
    st.number_input("UMAP min_dist", key="umap_min_dist", value=0.1, format="%.2f", min_value=0.0, max_value=0.99)

# Load file and compute embeddings if requested
if uploaded is not None:
    try:
        df = read_xlsx(uploaded.read())
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.session_state.state["df"] = df
    st.write("Sample queries:", df.head(8))

    # compute embeddings (or re-compute if user asked)
    if embed_button or st.session_state.state["embeddings"] is None:
        if not API_KEY:
            st.error("No API key configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY in environment or Streamlit secrets.")
        else:
            with st.spinner("Computing embeddings (this may take a little while)..."):
                texts = df["Queries"].tolist()
                try:
                    embeddings = compute_embeddings_openai(texts)
                except Exception as e:
                    st.error(f"Embedding API error: {e}")
                    st.stop()
                st.session_state.state["texts"] = texts
                st.session_state.state["embeddings"] = embeddings
                # compute umap now using the UI controls
                umap_2d = compute_umap(
                    embeddings,
                    n_neighbors=int(st.session_state.get("umap_neighbors", 15)),
                    min_dist=float(st.session_state.get("umap_min_dist", 0.1))
                )
                st.session_state.state["umap_2d"] = umap_2d
                st.session_state.state["similarity"] = make_similarity_matrix(embeddings)
                st.success("Embeddings computed and cached.")
    else:
        st.info("Using cached embeddings for this session. Click 'Compute embeddings' to re-embed.")

# If we have embeddings, do clustering and visualizations
if st.session_state.state["embeddings"] is not None:
    df = st.session_state.state["df"]
    embeddings = st.session_state.state["embeddings"]
    texts = st.session_state.state["texts"]
    umap_2d = st.session_state.state["umap_2d"]
    similarity = st.session_state.state["similarity"]

    # Cluster (KMeans)
    labels, kmodel = cluster_kmeans(embeddings, k)
    st.session_state.state["clusters"] = labels
    df["cluster"] = labels
    df["umap_x"] = umap_2d[:, 0]
    df["umap_y"] = umap_2d[:, 1]

    # Cluster summary
    st.markdown("### Cluster summary")
    cluster_summary = df.groupby("cluster").agg(
        size=("Queries", "count"),
        sample=("Queries", lambda x: "; ".join(x.iloc[:3]))
    ).reset_index().sort_values("cluster")
    st.dataframe(cluster_summary)

    # UMAP scatter (Plotly)
    st.markdown("### UMAP 2D semantic map (interactive)")
    fig = px.scatter(df, x="umap_x", y="umap_y", color=df["cluster"].astype(str),
                     hover_data=["Queries"],
                     title="UMAP Projection colored by cluster")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Show items in selected cluster
    sel_cluster = st.selectbox("Inspect cluster", options=sorted(df["cluster"].unique().tolist()))
    st.write(df[df["cluster"] == sel_cluster][["Queries"]].reset_index(drop=True))

    # Similarity table for a selected query
    sel_idx = st.number_input("Query index for similarity lookup (0-based)", min_value=0, max_value=len(texts) - 1, value=0, step=1)
    if st.button("Show top similar queries"):
        sims = similarity[sel_idx]
        top_idx = np.argsort(-sims)[1:11]  # excluding self
        out = pd.DataFrame({
            "index": top_idx,
            "query": [texts[i] for i in top_idx],
            "score": [float(sims[i]) for i in top_idx],
            "cluster": [int(df.iloc[i]["cluster"]) for i in top_idx]
        })
        st.write("Base query:", texts[sel_idx])
        st.dataframe(out)

    # Similarity heatmap
    if st.session_state.show_sim:
        st.markdown("### Similarity matrix (first 200 queries shown)")
        max_show = min(len(texts), 200)
        fig2 = go.Figure(data=go.Heatmap(
            z=similarity[:max_show, :max_show],
            x=[f"{i}" for i in range(max_show)],
            y=[f"{i}" for i in range(max_show)],
            colorbar=dict(title="Cosine")
        ))
        fig2.update_layout(height=700, title="Cosine similarity (showing indices)")
        st.plotly_chart(fig2, use_container_width=True)

    # Dendrogram
    if st.session_state.show_dend:
        st.markdown("### Hierarchical dendrogram (Ward linkage)")
        npoints = min(len(embeddings), 250)
        sample_idx = np.linspace(0, len(embeddings) - 1, npoints, dtype=int)
        sub_emb = embeddings[sample_idx]
        Z = linkage(sub_emb, method="ward")
        fig3 = plt.figure(figsize=(12, 6))
        dendrogram(Z, labels=[str(i) for i in sample_idx], leaf_rotation=90, leaf_font_size=8)
        st.pyplot(fig3)

    # Network graph (pyvis) - show cluster-level network of similarities
    if st.session_state.show_net:
        st.markdown("### Network (top similarity edges)")

        N_nodes = min(150, len(texts))
        nodes_idx = [int(i) for i in range(N_nodes)]

        net = Network(height="700px", width="100%", directed=False, notebook=False)
        net.barnes_hut()

        sim_sub = similarity[np.ix_(nodes_idx, nodes_idx)]

        # Add nodes
        for i in nodes_idx:
            label_text = texts[i][:60] + "..." if len(texts[i]) > 60 else texts[i]
            net.add_node(int(i), label=label_text, title=texts[i])

        # Add edges (top N similarity links)
        for i in nodes_idx:
            topn = np.argsort(-sim_sub[i])[1:4]
            for j in topn:
                j = int(j)
                score = float(sim_sub[i, j])
                if score > 0.40:   # threshold for readability
                    net.add_edge(int(i), int(j), value=score)

        path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
        net.write_html(path, local=True)

        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        st.components.v1.html(html, height=720, scrolling=True)


    st.markdown("---")
    st.markdown("## SEO-relevant outputs & suggestions")
    st.markdown("""
    1. **Per-cluster page mapping** — treat each cluster as candidate *topic groups*. If cluster size is large (>N queries) you might:
       - consolidate into a pillar page + multiple supporting pages (internal linking).
       - if queries in cluster are very diverse (low average similarity) split into subclusters.
    2. **Similarity scores** let you detect near-duplicates (score > 0.9) — merge or canonicalize them.
    3. **Topical authority decision**: compute average inter-cluster similarity. If clusters are very distant (avg inter-cluster cosine < 0.2) then a single domain may struggle to rank for both topics; consider separate domains/brand personalities for very distant business offerings.
    4. **Follow-up**: for each cluster, extract top tokens / keywords (simple TF-IDF on the query texts) and use them as page title/meta ideas.
    """)

    # Provide downloadable CSV with clusters + scores
    tmp = df.copy()
    tmp["similarity_to_centroid"] = [float(cosine_similarity([embeddings[i]], [kmodel.cluster_centers_[labels[i]]])[0, 0]) for i in range(len(embeddings))]
    csv = tmp.to_csv(index=False).encode('utf-8')
    st.download_button("Download cluster CSV", csv, file_name="queries_with_clusters.csv", mime="text/csv")

else:
    st.info("Upload a .xlsx and compute embeddings to start.")

# End of streamlit_app.py
