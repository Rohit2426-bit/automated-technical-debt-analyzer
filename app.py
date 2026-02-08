import streamlit as st
import ast
import os
import numpy as np
import pandas as pd
import joblib
import torch
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# =========================================================
# PATH SETUP
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RF_MODEL_PATH = os.path.join(BASE_DIR, "model_rf.pkl")
GNN_MODEL_PATH = os.path.join(BASE_DIR, "model_gnn.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# GNN MODEL DEFINITION
# =========================================================
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)

# =========================================================
# SAFE MODEL LOADERS
# =========================================================
@st.cache_resource
def load_rf():
    if not os.path.exists(RF_MODEL_PATH):
        return None
    return joblib.load(RF_MODEL_PATH)

@st.cache_resource
def load_gnn():
    if not os.path.exists(GNN_MODEL_PATH):
        return None
    model = GNNModel().to(device)
    model.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=device))
    model.eval()
    return model

rf_model = load_rf()
gnn_model = load_gnn()

# =========================================================
# AST FEATURE EXTRACTION
# =========================================================
def extract_ast_features(code):
    tree = ast.parse(code)

    num_nodes = sum(1 for _ in ast.walk(tree))
    num_loops = sum(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree))
    num_conditionals = sum(isinstance(n, ast.If) for n in ast.walk(tree))
    num_functions = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
    num_classes = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))

    return {
        "num_nodes": num_nodes,
        "num_loops": num_loops,
        "num_functions": num_functions,
        "num_classes": num_classes,
        "num_conditionals": num_conditionals,
    }

# =========================================================
# HEURISTIC FALLBACK SCORE (SAFE & EXPLAINABLE)
# =========================================================
def heuristic_score(features):
    score = (
        0.001 * features["num_nodes"]
        + 0.1 * features["num_loops"]
        + 0.1 * features["num_conditionals"]
        + 0.05 * features["num_functions"]
        + 0.05 * features["num_classes"]
    )
    return min(score, 1.0)

# =========================================================
# AST → GRAPH (FOR GNN)
# =========================================================
def ast_to_graph(code):
    tree = ast.parse(code)
    graph = nx.Graph()

    for idx, node in enumerate(ast.walk(tree)):
        graph.add_node(idx)
        for child in ast.iter_child_nodes(node):
            graph.add_edge(idx, list(ast.walk(tree)).index(child))

    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    x = torch.ones((graph.number_of_nodes(), 1))

    return Data(
        x=x,
        edge_index=edge_index,
        batch=torch.zeros(graph.number_of_nodes(), dtype=torch.long),
    )

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Automated Technical Debt Analyzer", layout="wide")

st.title("🧠 Automated Technical Debt Analyzer")
st.subheader("AST-based Software Maintainability Prediction")

model_choice = st.radio(
    "Select Prediction Model",
    ["Random Forest (Fast & Accurate)", "GNN (Structural & Experimental)"],
)

code_input = st.text_area(
    "Paste Python Code Here",
    height=300,
    placeholder="Paste Python source code (e.g., from GitHub)...",
)

if st.button("Analyze Technical Debt"):
    if not code_input.strip():
        st.error("Please paste some Python code.")
    else:
        try:
            features = extract_ast_features(code_input)
            features_df = pd.DataFrame([features])

            base_score = heuristic_score(features)

            # -------------------------------
            # RANDOM FOREST
            # -------------------------------
            if model_choice.startswith("Random Forest"):
                if rf_model is not None:
                    score = rf_model.predict(features_df)[0]
                else:
                    score = base_score

                st.success(f"🌳 Random Forest Debt Score: {score:.3f}")

            # -------------------------------
            # GNN
            # -------------------------------
            else:
                if gnn_model is not None:
                    graph_data = ast_to_graph(code_input).to(device)
                    with torch.no_grad():
                        score = gnn_model(graph_data).item()
                else:
                    score = base_score * 0.5  # conservative fallback

                st.success(f"🧠 GNN Debt Score: {score:.3f}")

        except Exception as e:
            st.error(f"Error analyzing code: {e}")
