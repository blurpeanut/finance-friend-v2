import streamlit as st
import graphviz

st.set_page_config(page_title="Methodology", layout="wide")
st.title("🔍 Methodology")

st.markdown("""
Welcome to the Methodology page. Here, we explain the data flows and implementation logic behind Finance Friend's two core use cases:

1. **Intelligent Q&A Chat with Policy Documents**
2. **Interactive Budget Visualisation**

Each use case is supported by a flowchart and a description of how the backend and frontend systems work together.
""")

st.header("🧠 Use Case A: Policy Q&A with LLM")

rag_flow = graphviz.Digraph("RAG_Flow", format="png")
rag_flow.attr(rankdir="TB", splines="ortho", fontsize="10")

# Nodes for RAG Flow
rag_flow.node("U", "User submits a question", shape="ellipse", style="filled", fillcolor="#E6F2FF")
rag_flow.node("E", "Embed question (OpenAI Embedding)", shape="box")
rag_flow.node("VS", "Retrieve top chunks\nfrom Vector DB", shape="box")
rag_flow.node("CTX", "Build context:\n[Top Chunks + Question]", shape="box")
rag_flow.node("LLM", "LLM generates grounded answer", shape="box", style="filled", fillcolor="#FFF2CC")
rag_flow.node("OUT", "Display answer to user", shape="ellipse", style="filled", fillcolor="#E6F2FF")

# Document Upload Nodes
rag_flow.node("UPLOAD", "Admin uploads\nPolicy Documents", shape="parallelogram", style="filled", fillcolor="#D5E8D4")
rag_flow.node("CHUNK", "Chunk & Embed Documents", shape="box")
rag_flow.node("STORE", "Store embeddings in Vector DB", shape="box", style="filled", fillcolor="#D5E8D4")

# Flow edges
rag_flow.edges([
    ("UPLOAD", "CHUNK"),
    ("CHUNK", "STORE"),
    ("U", "E"),
    ("E", "VS"),
    ("VS", "CTX"),
    ("CTX", "LLM"),
    ("LLM", "OUT")
])
rag_flow.edge("STORE", "VS", label="retrieve")

st.graphviz_chart(rag_flow)

# ================
# USE CASE B: Budget Table + Visualisation (Read-only)
# ================

st.header("📊 Use Case B: Budget Visualisation and Read‑Only Table")

st.markdown("""
This feature ingests the latest GL file, cleans and standardises it, then visualises spending against category budgets.
The transaction table is **read-only**. Users can **filter** what they see and **export CSV**, but they cannot edit rows in-app.
""")

budget_flow = graphviz.Digraph("Budget_Flow_RO", format="png")
budget_flow.attr(rankdir="TB", splines="ortho", fontsize="10")

# Styling helpers
p_fill = {"shape": "parallelogram", "style": "filled", "fillcolor": "#D5E8D4"}   # inputs/uploads
b_fill = {"shape": "box"}                                                         # processes
o_fill = {"shape": "ellipse", "style": "filled", "fillcolor": "#E6F2FF"}          # outputs/UI
y_fill = {"shape": "box", "style": "filled", "fillcolor": "#FFF2CC"}              # calc/metrics

# Nodes
budget_flow.node("B1", "Admin uploads GL file\n(.xlsx or .csv)", **p_fill)
budget_flow.node("B2", "Validate schema & clean data\n(trim headers, types, dates, amounts)", **b_fill)
budget_flow.node("B3", "Standardise fields\n(Directorate, Vote Type, Categories)", **b_fill)
budget_flow.node("B4", "Compute budget utilisation\nper category", **y_fill)
budget_flow.node("B5", "Render budget bar chart", **o_fill)
budget_flow.node("B6", "Read‑only transaction table", **o_fill)

# User interactions (not editing)
budget_flow.node("B7", "User applies filters\n(Directorate / Vote Type / Date)", **b_fill)
budget_flow.node("B8", "Export current view\nas CSV", **p_fill)

# Edges
budget_flow.edges([
    ("B1", "B2"),
    ("B2", "B3"),
    ("B3", "B4"),
    ("B4", "B5"),
    ("B3", "B6"),   # table is derived from cleaned/standardised data
    ("B7", "B5"),   # filters adjust chart
    ("B7", "B6"),   # filters adjust table
    ("B6", "B8")    # export from current table view
])

st.graphviz_chart(budget_flow)

# ================
# Summary
# ================

st.header("🧾 Summary")

st.markdown("""
- The **RAG pipeline** empowers users to ask open-ended questions grounded in official policy documents.
- The **Budget visualisation tool** enables clear tracking of expenditure against caps, with real-time interaction and visualisation.
- Both features are designed to be user-friendly, traceable, and easy to maintain.
""")
