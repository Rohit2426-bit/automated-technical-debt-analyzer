import streamlit as st
import ast
import pandas as pd
import joblib
import torch
import networkx as nx
import torch.nn.functional as F
import os

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# =====================================================
# PATH FIX
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RF_MODEL_PATH = os.path.join(BASE_DIR, "model_rf.pkl")
GNN_MODEL_PATH = os.path.join(BASE_DIR, "model_gnn.pt")

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Automated Technical Debt Analyzer",
    layout="wide"
)

# =====================================================
# GNN MODEL
# =====================================================
class GNNRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 32)
        self.conv2 = GCNConv(32, 64)
        self.lin = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_rf():
    return joblib.load(RF_MODEL_PATH)

@st.cache_resource
def load_gnn():
    device = torch.device("cpu")
    model = GNNRegressor().to(device)
    model.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=device))
    model.eval()
    return model

rf_model = load_rf()
gnn_model = load_gnn()

# =====================================================
# AST FEATURE EXTRACTOR (RF)
# =====================================================
class ASTFeatureExtractor(ast.NodeVisitor):
    def __init__(self):
        self.num_nodes = 0
        self.num_loops = 0
        self.num_functions = 0
        self.num_classes = 0
        self.num_conditionals = 0

    def generic_visit(self, node):
        self.num_nodes += 1
        super().generic_visit(node)

    def visit_For(self, node):
        self.num_loops += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.num_loops += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.num_functions += 1
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.num_classes += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.num_conditionals += 1
        self.generic_visit(node)

# =====================================================
# AST → GRAPH (GNN)
# =====================================================
NODE_TYPES = {}

def encode_node_type(node_type):
    if node_type not in NODE_TYPES:
        NODE_TYPES[node_type] = len(NODE_TYPES)
    return NODE_TYPES[node_type]

def ast_to_graph(tree):
    G = nx.DiGraph()
    for node in ast.walk(tree):
        G.add_node(id(node), node_type=type(node).__name__)
        for child in ast.iter_child_nodes(node):
            G.add_edge(id(node), id(child))
    return G

def graph_to_pyg(G):
    node_map = {}
    x = []

    for i, (n, attrs) in enumerate(G.nodes(data=True)):
        node_map[n] = i
        x.append([encode_node_type(attrs["node_type"])])

    edge_index = [[node_map[u], node_map[v]] for u, v in G.edges()]

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        batch=torch.zeros(len(x), dtype=torch.long)
    )

# =====================================================
# UI
# =====================================================
st.title("🧠 Automated Technical Debt Analyzer")
st.subheader("AST-based Software Maintainability Prediction")

st.markdown(
    "Paste Python source code below and select a model to "
    "estimate technical debt severity."
)

model_choice = st.radio(
    "Select Prediction Model",
    ["Random Forest (Fast & Accurate)", "GNN (Structural & Experimental)"]
)

code_input = st.text_area(
    "Paste Python Code Here",
    height=300,
    placeholder="def example():\n    for i in range(10):\n        if i % 2 == 0:\n            print(i)"
)

# =====================================================
# PREDICTION
# =====================================================
if st.button("Analyze Technical Debt"):
    try:
        tree = ast.parse(code_input)

        if model_choice.startswith("Random"):
            extractor = ASTFeatureExtractor()
            extractor.visit(tree)

            features = pd.DataFrame([{
                "num_nodes": extractor.num_nodes,
                "num_loops": extractor.num_loops,
                "num_functions": extractor.num_functions,
                "num_classes": extractor.num_classes,
                "num_conditionals": extractor.num_conditionals
            }])

            score = rf_model.predict(features)[0]
            st.success(f"🌳 Random Forest Debt Score: {score:.3f}")

        else:
            G = ast_to_graph(tree)
            data = graph_to_pyg(G)

            with torch.no_grad():
                score = gnn_model(data).item()

            st.success(f"🧠 GNN Debt Score: {score:.3f}")

    except SyntaxError:
        st.error("❌ Invalid Python syntax.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
