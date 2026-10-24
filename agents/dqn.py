"""Double DQN written from scratch (PyTorch)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, obs_dim, capacity=100_000):
        self.cap, self.n, self.i = capacity, 0, 0
        self.o = np.zeros((capacity, obs_dim), np.float32)
        self.o2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)

    def add(self, o, a, r, o2, d):
        i = self.i
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch):
        idx = np.random.randint(0, self.n, size=batch)
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx]

    def __len__(self):
        return self.n


class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=5e-4, gamma=0.95, batch=128,
                 tau=0.005, buffer=100_000, noop_explore=0.7, device="cpu"):
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.gamma, self.batch, self.tau = gamma, batch, tau
        self.noop_explore = noop_explore
        self.device = torch.device(device)
        self.q = QNet(obs_dim, n_actions).to(self.device)
        self.q_targ = QNet(obs_dim, n_actions).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(obs_dim, buffer)

    @torch.no_grad()
    def act(self, obs, eps=0.0):
        if np.random.rand() < eps:
            # biased exploration: most random moves are no-ops (otherwise it churns the table every step)
            if np.random.rand() < self.noop_explore:
                return 0
            return int(np.random.randint(1, self.n_actions))
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.q(x).argmax(dim=1).item())

    def update(self):
        if len(self.buffer) < self.batch:
            return None
        o, a, r, o2, d = [torch.as_tensor(x, device=self.device) for x in self.buffer.sample(self.batch)]
        with torch.no_grad():
            a_star = self.q(o2).argmax(dim=1, keepdim=True)              # Double DQN: online net selects
            q_next = self.q_targ(o2).gather(1, a_star).squeeze(1)        # target net evaluates
            y = r + self.gamma * (1.0 - d) * q_next
        q_sa = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_targ.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
        return float(loss.item())

    def save(self, path, cfg_dict, tag):
        torch.save({"state_dict": self.q.state_dict(), "obs_dim": self.obs_dim,
                    "n_actions": self.n_actions, "cfg": cfg_dict, "tag": tag}, path)

    @classmethod
    def from_checkpoint(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(ck["obs_dim"], ck["n_actions"])
        agent.q.load_state_dict(ck["state_dict"])
        agent.q_targ.load_state_dict(ck["state_dict"])
        return agent, ck
