"""Common architectures with no torchvision or transformers implementation, written out in plain PyTorch."""
import torch
import torch.nn as nn

from .inputs import ATARI_FRAMES, CRITEO_DENSE, CRITEO_SPARSE, GAN_NOISE, RGB


class UNet(nn.Module):
    LEVELS = 4

    def __init__(self, base, in_channels, classes):
        super().__init__()
        widths = [base * 2 ** i for i in range(self.LEVELS)]

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.down = nn.ModuleList()
        cin = in_channels
        for w in widths:
            self.down.append(block(cin, w))
            cin = w
        self.bottom = block(widths[-1], widths[-1] * 2)
        self.up = nn.ModuleList()
        self.merge = nn.ModuleList()
        cin = widths[-1] * 2
        for w in reversed(widths):
            self.up.append(nn.ConvTranspose2d(cin, w, 2, stride=2))
            self.merge.append(block(w * 2, w))
            cin = w
        self.head = nn.Conv2d(widths[0], classes, 1)

    def forward(self, x):
        skips = []
        for down in self.down:
            x = down(x)
            skips.append(x)
            x = nn.functional.max_pool2d(x, 2)
        x = self.bottom(x)
        for up, merge, skip in zip(self.up, self.merge, reversed(skips)):
            x = merge(torch.cat([up(x), skip], dim=1))
        return self.head(x)


class DCGANGenerator(nn.Module):
    # Noise vector to a 64x64 image.
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(GAN_NOISE, width * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(width * 8), nn.ReLU(True),
            nn.ConvTranspose2d(width * 8, width * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(width * 4), nn.ReLU(True),
            nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(width * 2), nn.ReLU(True),
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(True),
            nn.ConvTranspose2d(width, RGB, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class MLP(nn.Module):
    def __init__(self, inputs, width, depth, outputs):
        super().__init__()
        layers = []
        for i in range(depth):
            layers += [nn.Linear(inputs if i == 0 else width, width), nn.ReLU()]
        self.net = nn.Sequential(*layers, nn.Linear(width, outputs))

    def forward(self, x):
        return self.net(x)


class LSTMForecaster(nn.Module):
    # Reads a window of a multivariate series and predicts the next step.
    # The cell is written out with nn.Linear because FlopCounterMode can't count nn.LSTM's fused kernel.
    def __init__(self, channels, hidden, layers):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.ModuleList(nn.Linear((channels if i == 0 else hidden) + hidden, 4 * hidden) for i in range(layers))
        self.head = nn.Linear(hidden, channels)

    def forward(self, x):
        for gates in self.gates:
            h = x.new_zeros(x.shape[0], self.hidden)
            c = x.new_zeros(x.shape[0], self.hidden)
            outputs = []
            for t in range(x.shape[1]):
                i, f, g, o = gates(torch.cat([x[:, t], h], dim=1)).chunk(4, dim=1)
                c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
                h = torch.sigmoid(o) * torch.tanh(c)
                outputs.append(h)
            x = torch.stack(outputs, dim=1)
        return self.head(x[:, -1])


class GraphNet(nn.Module):
    # Node classification over a fixed random graph.
    # GCN: average the neighbours, then a linear layer. GIN: sum the neighbours, then a 2-layer MLP.
    def __init__(self, kind, nodes, edges, inputs, hidden, layers, classes):
        super().__init__()
        self.kind = kind
        src = torch.randint(0, nodes, (edges,))
        dst = torch.randint(0, nodes, (edges,))
        self_loops = torch.arange(nodes)
        self.register_buffer("src", torch.cat([src, dst, self_loops]))
        self.register_buffer("dst", torch.cat([dst, src, self_loops]))
        self.register_buffer("degree", torch.bincount(self.dst, minlength=nodes).float().unsqueeze(1))
        dims = [inputs] + [hidden] * (layers - 1) + [classes]
        if kind == "gcn":
            self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        else:
            self.layers = nn.ModuleList(
                nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, b)) for a, b in zip(dims, dims[1:]))

    def forward(self, x):
        x = x[0]
        for i, layer in enumerate(self.layers):
            agg = torch.zeros_like(x).index_add_(0, self.dst, x[self.src])
            if self.kind == "gcn":
                agg = agg / self.degree
            x = layer(agg)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class PointNet(nn.Module):
    # Classifies a point cloud: shared per-point MLP, max-pool, then a classifier.
    def __init__(self, classes):
        super().__init__()
        self.point = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.head = MLP(inputs=1024, width=512, depth=2, outputs=classes)

    def forward(self, points):
        return self.head(self.point(points).max(dim=2).values)


class DLRM(nn.Module):
    # Click prediction on Criteo-style rows: one embedding table per categorical field, plus two MLPs.
    def __init__(self, rows, dim):
        super().__init__()
        self.tables = nn.ModuleList(nn.Embedding(rows, dim) for _ in range(CRITEO_SPARSE))
        self.bottom = MLP(inputs=CRITEO_DENSE, width=512, depth=2, outputs=dim)
        self.top = MLP(inputs=dim * (CRITEO_SPARSE + 1), width=1024, depth=3, outputs=1)

    def forward(self, x):
        dense = self.bottom(x[:, :CRITEO_DENSE])
        ids = x[:, CRITEO_DENSE:].long()
        sparse = [table(ids[:, i]) for i, table in enumerate(self.tables)]
        return self.top(torch.cat([dense] + sparse, dim=1))


class AtariPolicy(nn.Module):
    # The DQN "Nature" network over a stack of 84x84 frames.
    def __init__(self, actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ATARI_FRAMES, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, actions),
        )

    def forward(self, x):
        return self.net(x)
