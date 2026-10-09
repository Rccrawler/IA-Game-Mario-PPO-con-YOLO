"""
PPO en PyTorch + percepción por visión (YOLO entrenado por ti).

El agente SOLO recibe imágenes: píxeles + objetos que YOLO detecta en ellos.
No se usa ningún dato interno del juego.

Uso:
    python mario_ppo_vision.py test    # comprueba que la percepción funciona
    python mario_ppo_vision.py         # entrena
"""
import glob
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_weights():
    """Busca el best.pt más reciente generado por el entrenamiento."""
    files = glob.glob("runs/**/mario_detector*/weights/best.pt", recursive=True)
    if not files:
        raise FileNotFoundError("No encuentro best.pt. ¿Terminó el entrenamiento de YOLO?")
    return max(files, key=os.path.getmtime)


# ---------------------------------------------------------------
# 1. PERCEPCIÓN: imagen -> detecciones -> vector de tamaño fijo
# ---------------------------------------------------------------
class Perception:
    """
    Vector resultante:
      [mario_visible, mario_x, mario_y]
      + por cada otra clase, K objetos más cercanos a Mario:
        [presente, dx, dy]  (dx, dy relativos a Mario, normalizados a [-1, 1])
    """

    def __init__(self, weights, K=4, conf=0.4, exclude=()):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        names = self.model.names  # {id: nombre}
        self.mario_id = next(i for i, n in names.items() if n == "mario")
        self.other_ids = sorted(i for i in names
                                if i != self.mario_id and names[i] not in exclude)
        self.class_names = [names[i] for i in self.other_ids]
        self.K, self.conf = K, conf
        self.dim = 3 + len(self.other_ids) * K * 3
        self.last_mario = (100.0, 180.0)  # por si Mario no se detecta

    def detect(self, frame_rgb):
        # Ultralytics espera BGR en arrays de numpy (como OpenCV)
        bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
        r = self.model.predict(bgr, imgsz=256, conf=self.conf, verbose=False)[0]
        xywh = r.boxes.xywh.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        score = r.boxes.conf.cpu().numpy()
        return xywh, cls, score

    def __call__(self, frame_rgb):
        H, W = frame_rgb.shape[:2]
        xywh, cls, score = self.detect(frame_rgb)
        self.last_det = (xywh, cls, score)
        feat = np.zeros(self.dim, dtype=np.float32)

        # Mario: la detección con más confianza
        m = np.where(cls == self.mario_id)[0]
        if len(m):
            best = m[np.argmax(score[m])]
            self.last_mario = (float(xywh[best][0]), float(xywh[best][1]))
            feat[0] = 1.0
        mx, my = self.last_mario
        feat[1], feat[2] = mx / W, my / H

        # Resto de clases: K más cercanos a Mario
        pos = 3
        for cid in self.other_ids:
            idx = np.where(cls == cid)[0]
            objs = [(xywh[i][0] - mx, xywh[i][1] - my) for i in idx]
            objs.sort(key=lambda d: d[0] ** 2 + d[1] ** 2)
            for k in range(self.K):
                if k < len(objs):
                    dx, dy = objs[k]
                    feat[pos:pos + 3] = [1.0, np.clip(dx / W, -1, 1), np.clip(dy / H, -1, 1)]
                pos += 3
        return feat


# ---------------------------------------------------------------
# 2. ENTORNO: devuelve (píxeles apilados, vector de percepción)
# ---------------------------------------------------------------
class MarioVisionEnv:
    def __init__(self, perception, skip=4, stack=4, world_stage="1-1"):
        name = f"SuperMarioBros-{world_stage}-v0"
        self.env = JoypadSpace(gym_super_mario_bros.make(name), SIMPLE_MOVEMENT)
        self.perception = perception
        self.skip, self.stack = skip, stack
        self.frames = deque(maxlen=stack)
        self.n_actions = len(SIMPLE_MOVEMENT)

    @staticmethod
    def _proc(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self):
        out = self.env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        f = self._proc(obs)
        for _ in range(self.stack):
            self.frames.append(f)
        return np.stack(self.frames), self.perception(obs)

    def step(self, action):
        total, done, info = 0.0, False, {}
        for _ in range(self.skip):
            out = self.env.step(action)
            if len(out) == 5:
                obs, r, term, trunc, info = out
                done = term or trunc
            else:
                obs, r, done, info = out
            total += r
            if done:
                break
        self.frames.append(self._proc(obs))
        if info.get("flag_get"):
            total += 50
        # YOLO se ejecuta una vez por paso del agente, sobre el frame completo
        return (np.stack(self.frames), self.perception(obs)), total / 10.0, done, info


# ---------------------------------------------------------------
# 3. RED: rama de píxeles (CNN) + rama de objetos (MLP) -> actor/crítico
# ---------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, n_actions, feat_dim, in_channels=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
        )
        self.obj_mlp = nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU())
        self.actor = nn.Linear(512 + 128, n_actions)
        self.critic = nn.Linear(512 + 128, 1)

    def forward(self, pixels, feats):
        h = torch.cat([self.cnn(pixels.float() / 255.0), self.obj_mlp(feats)], dim=1)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, pixels, feats):
        logits, value = self(pixels, feats)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), value


# ---------------------------------------------------------------
# 4. PPO
# ---------------------------------------------------------------
def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_v = last_value if t == len(rewards) - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * mask - values[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
    return adv, adv + values


def to_t(x):
    return torch.as_tensor(x, device=DEVICE).unsqueeze(0)


def train(total_steps=1_000_000, rollout=1024, epochs=4, batch=256,
          lr=2.5e-4, clip=0.2, vf_coef=0.5, ent_coef=0.01):
    perception = Perception(find_weights())
    env = MarioVisionEnv(perception)
    model = ActorCritic(env.n_actions, perception.dim).to(DEVICE)
    if os.path.exists("mario_ppo_vision.pt"):
        model.load_state_dict(torch.load("mario_ppo_vision.pt", map_location=DEVICE))
        print("Reanudando desde mario_ppo_vision.pt")
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    (pix, ft) = env.reset()
    ep_reward, ep_rewards, steps = 0.0, deque(maxlen=10), 0

    while steps < total_steps:
        P, F, A, LP, R, D, V = [], [], [], [], [], [], []
        for _ in range(rollout):
            p, f = to_t(pix), to_t(ft)
            with torch.no_grad():
                a, lp, v = model.act(p, f)
            (pix2, ft2), r, done, _ = env.step(a.item())

            P.append(p.squeeze(0)); F.append(f.squeeze(0)); A.append(a.squeeze(0))
            LP.append(lp.squeeze(0)); R.append(r); D.append(float(done)); V.append(v.squeeze(0))

            ep_reward += r
            pix, ft = pix2, ft2
            if done:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                pix, ft = env.reset()
        steps += rollout

        P, F, A, LP, V = map(torch.stack, (P, F, A, LP, V))
        R = torch.tensor(R, device=DEVICE, dtype=torch.float32)
        D = torch.tensor(D, device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            _, last_v = model(to_t(pix), to_t(ft))
        adv, ret = compute_gae(R, V, D, last_v.squeeze(0))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = np.arange(rollout)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for i in range(0, rollout, batch):
                b = idx[i:i + batch]
                logits, value = model(P[b], F[b])
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(A[b]) - LP[b]).exp()
                s1 = ratio * adv[b]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[b]
                loss = (-torch.min(s1, s2).mean()
                        + vf_coef * (ret[b] - value).pow(2).mean()
                        - ent_coef * dist.entropy().mean())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        mean_r = np.mean(ep_rewards) if ep_rewards else 0.0
        print(f"steps {steps:>9} | recompensa media (10 ep): {mean_r:8.2f}")
        torch.save(model.state_dict(), "mario_ppo_vision.pt")


def test():
    """Juega al azar unos pasos y enseña lo que ve YOLO, sin entrenar."""
    perception = Perception(find_weights())
    env = MarioVisionEnv(perception)
    env.reset()
    for t in range(40):
        (pix, ft), _, done, _ = env.step(np.random.choice([1, 2, 3, 4]))
        if t % 10 == 0:
            print(f"paso {t}: Mario visible={bool(ft[0])}, "
                  f"objetos detectados={int(ft[3::3].sum())}")
        if done:
            env.reset()
    print(f"Dimensión del vector de percepción: {perception.dim}")
    print("Clases:", perception.class_names)


if __name__ == "__main__":
    test() if (len(sys.argv) > 1 and sys.argv[1] == "test") else train()
