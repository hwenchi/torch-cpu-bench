from transformers import PatchTSTConfig, PatchTSTModel

from .custom import DLRM, GraphNet, LSTMForecaster, MLP, AtariPolicy, PointNet
from .inputs import (
    ATARI_FRAMES, ATARI_SIZE, CORA_EDGES, CORA_FEATURES, CORA_NODES, POINTS, SERIES_CHANNELS, SERIES_LEN, click_log,
    features,
)

CORA_CLASSES = 7
MODELNET_CLASSES = 40
ATARI_ACTIONS = 18
DLRM_ROWS = 100_000

cora = features(CORA_NODES, CORA_FEATURES, label=f"graph of {CORA_NODES:,} nodes, {CORA_FEATURES:,} features each")

MODELS = [
    {"name": "mlp_small", "family": "MLP", "task": "Tabular",
     "build": lambda: MLP(inputs=64, width=256, depth=3, outputs=1), "input": features(64, label="row of 64 features")},
    {"name": "mlp_large", "family": "MLP", "task": "Tabular",
     "build": lambda: MLP(inputs=256, width=2048, depth=4, outputs=1), "input": features(256, label="row of 256 features")},

    {"name": "lstm_forecaster", "family": "LSTM", "task": "Time series",
     "build": lambda: LSTMForecaster(channels=32, hidden=256, layers=2), "input": features(128, 32, label="128 time steps of 32 channels")},
    {"name": "patchtst", "family": "PatchTST", "task": "Time series",
     "build": lambda: PatchTSTModel(PatchTSTConfig(num_input_channels=SERIES_CHANNELS, context_length=SERIES_LEN)),
     "input": features(SERIES_LEN, SERIES_CHANNELS, label=f"{SERIES_LEN} time steps of {SERIES_CHANNELS} channels")},

    {"name": "gcn_cora", "family": "GNN", "task": "Graph",
     "build": lambda: GraphNet("gcn", nodes=CORA_NODES, edges=CORA_EDGES, inputs=CORA_FEATURES, hidden=64, layers=2, classes=CORA_CLASSES),
     "input": cora},
    {"name": "gin_cora", "family": "GNN", "task": "Graph",
     "build": lambda: GraphNet("gin", nodes=CORA_NODES, edges=CORA_EDGES, inputs=CORA_FEATURES, hidden=64, layers=3, classes=CORA_CLASSES),
     "input": cora},

    {"name": "pointnet", "family": "PointNet", "task": "Point cloud",
     "build": lambda: PointNet(classes=MODELNET_CLASSES), "input": features(3, POINTS, label=f"{POINTS:,} 3D points")},

    {"name": "dlrm_small", "family": "DLRM", "task": "Recommendation",
     "build": lambda: DLRM(rows=DLRM_ROWS, dim=64), "input": click_log(DLRM_ROWS)},

    {"name": "dqn_atari", "family": "DQN", "task": "Reinforcement learning",
     "build": lambda: AtariPolicy(actions=ATARI_ACTIONS), "input": features(ATARI_FRAMES, ATARI_SIZE, ATARI_SIZE, label=f"{ATARI_FRAMES} stacked {ATARI_SIZE}×{ATARI_SIZE} game frames")},
]
