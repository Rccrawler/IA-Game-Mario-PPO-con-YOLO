"""
Agente PPO que SOLO ve lo que detecta YOLO (sin píxeles), con memoria GRU.

Qué ve el agente en cada paso (todo sale de las cajas de YOLO):
  - Mario: visible, posición, ANCHO y ALTO.
  - Por cada clase, los K=4 objetos más cercanos: presente, dx, dy (relativos
    a Mario), ANCHO y ALTO.
  - Un MAPA 2D de lo sólido alrededor de Mario (celdas de 8 px; de -32 a +160 px
    en x y de -48 a +48 px en y): suelo, tuberías y bloques si YOLO los detecta.
  - Velocidad de Mario y su última acción.
Además, como antes: peligro por acción y modelo de transición como tareas auxiliares.

Requiere mario_ppo_vision.py en la misma carpeta (reutiliza Perception).

    python mario_ppo_yolo8.py          # entrenar (Ctrl+C para parar, reanuda solo)
    python mario_ppo_yolo8.py play     # verlo jugar (arriba a la derecha: el mapa que ve)
    python mario_ppo_yolo8.py eval     # medir distancia real sin ventana
"""
import os
import sys
from collections import Counter, deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.distributions import Categorical

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from mario_ppo_vision import Perception, find_weights, compute_gae, DEVICE

SAVE_PATH = "mario_ppo_yolo8.pt"            # archivo nuevo: empieza de cero a propósito
STEPS_PATH = "mario_ppo_yolo8.steps"
REWARD_SCALE = 100.0   # 1 px de avance ~ 0.01 de recompensa
DEATH_PENALTY = 1.0
WIN_BONUS = 10.0
DANGER_HORIZON = 20

ACTION_NAMES = ["quieto", "der", "der+salto", "der+correr",
                "der+salto+correr", "salto", "izq"]

# >>> PURE-START  (funciones sin torch, fáciles de probar)
W, H = 256, 240
K = 4                                        # objetos por clase
SOLID_CLASSES = ("ground", "block", "brick", "stair", "pipe")  # ocupan espacio sólido
TOKEN_EXCLUDE = ("ground", "block", "brick", "stair", "pipe")  # solo van en el mapa
CELL = 8
GRID_COLS, GRID_ROWS = 32, 28
GRID_X0, GRID_Y0 = -64, -112                 # esquina del mapa respecto al centro de Mario


def build_features(xywh, cls, score, mario_id, other_ids, last_mario, K=K):
    """Vector de objetos: [Mario(5)] + por clase K x [presente, dx, dy, ancho, alto]."""
    feat = np.zeros(5 + len(other_ids) * K * 5, dtype=np.float32)
    m = np.where(cls == mario_id)[0]
    if len(m):
        b = m[np.argmax(score[m])]
        last_mario = tuple(float(v) for v in xywh[b])      # (cx, cy, w, h)
        feat[0] = 1.0
    mx, my, mw, mh = last_mario
    feat[1:5] = [mx / W, my / H, mw / 32.0, mh / 32.0]
    pos = 5
    for cid in other_ids:
        idx = np.where(cls == cid)[0]
        objs = sorted(((xywh[i][0] - mx, xywh[i][1] - my, xywh[i][2], xywh[i][3])
                       for i in idx), key=lambda o: o[0] ** 2 + o[1] ** 2)
        for k in range(K):
            if k < len(objs):
                dx, dy, w, h = objs[k]
                feat[pos:pos + 5] = [1.0, np.clip(dx / 128.0, -2, 2),
                                     np.clip(dy / 120.0, -2, 2), w / 32.0, h / 32.0]
            pos += 5
    return feat, last_mario


def solid_grid(xywh, cls, solid_ids, mx, my):
    """Mapa 2D (filas x columnas) de lo sólido alrededor de Mario. 1 = ocupado."""
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    for (cx, cy, w, h), c in zip(xywh, cls):
        if int(c) not in solid_ids:
            continue
        c0 = int(round((cx - w / 2 - mx - GRID_X0) / CELL))
        c1 = int(round((cx + w / 2 - mx - GRID_X0) / CELL))
        r0 = int(round((cy - h / 2 - my - GRID_Y0) / CELL))
        r1 = int(round((cy + h / 2 - my - GRID_Y0) / CELL))
        c0, c1 = max(c0, 0), min(max(c1, c0 + 1), GRID_COLS)
        r0, r1 = max(r0, 0), min(max(r1, r0 + 1), GRID_ROWS)
        if c0 < c1 and r0 < r1:
            grid[r0:r1, c0:c1] = 1.0
    return grid
# <<< PURE-END


class RichPerception(Perception):
    """YOLO -> (vector de objetos con ancho/alto, mapa 2D de lo sólido)."""

    def __init__(self, weights, K=K, conf=0.4):
        super().__init__(weights, K=K, conf=conf, exclude=TOKEN_EXCLUDE)
        names = self.model.names
        self.solid_ids = {i for i, n in names.items() if n in SOLID_CLASSES}
        if not self.solid_ids:
            raise ValueError("YOLO no tiene ninguna clase sólida (ground, pipe, block...).")
        self.dim_obj = 5 + len(self.other_ids) * K * 5
        self.last_mario = (100.0, 180.0, 16.0, 16.0)

    def observe(self, frame_rgb):
        xywh, cls, score = self.detect(frame_rgb)
        self.last_det = (xywh, cls, score)
        feat, self.last_mario = build_features(xywh, cls, score, self.mario_id,
                                               self.other_ids, self.last_mario, self.K)
        grid = solid_grid(xywh, cls, self.solid_ids,
                          self.last_mario[0], self.last_mario[1])
        return feat, grid


# ---------------------------------------------------------------
# ENTORNO
# ---------------------------------------------------------------
class YoloOnlyEnv:
    def __init__(self, perception, skip=4, world_stage="1-1"):
        name = f"SuperMarioBros-{world_stage}-v0"
        self.env = JoypadSpace(gym_super_mario_bros.make(name), SIMPLE_MOVEMENT)
        self.perception, self.skip = perception, skip
        self.n_actions = len(SIMPLE_MOVEMENT)
        self.obj_dim = perception.dim_obj
        self.trans_dim = self.obj_dim + 2                 # objetos + velocidad
        self.base_dim = self.trans_dim + GRID_ROWS * GRID_COLS
        self.last_action = 0
        self.last_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        self.world_prev = None

    def _base(self, frame, true_vel=None):
        feat, grid = self.perception.observe(frame)
        self.last_grid = grid
        vel = np.zeros(2, dtype=np.float32) if true_vel is None else true_vel
        return np.concatenate([feat, vel, grid.ravel()]).astype(np.float32)

    def _obs(self, base_feat):
        onehot = np.zeros(self.n_actions, dtype=np.float32)
        onehot[self.last_action] = 1.0
        return np.concatenate([base_feat, onehot])

    def reset(self):
        out = self.env.reset()
        frame = out[0] if isinstance(out, tuple) else out
        self.world_prev = None
        self.prev_gray = None
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.last_action = 0
        return self._obs(self._base(frame))

    def step(self, action):
        total, done, info = 0.0, False, {}
        for _ in range(self.skip):
            out = self.env.step(action)
            if len(out) == 5:
                frame, r, term, trunc, info = out
                done = term or trunc
            else:
                frame, r, done, info = out
            total += r
            if done:
                break
        
        # --- CÁLCULO DE VELOCIDAD UNIVERSAL (PURA VISIÓN) ---
        # Convierte el frame actual a escala de grises
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mx, my = self.perception.last_mario[0], self.perception.last_mario[1]

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            self.world_prev = (mx, my)
            self.cam_x, self.cam_y = 0.0, 0.0

        # cv2.phaseCorrelate encuentra el desplazamiento (dx, dy) entre dos imágenes.
        # Recortamos los primeros 48 píxeles (HUD estático) para que no engañe al cálculo.
        crop = 48
        shift, _ = cv2.phaseCorrelate(
            np.float32(self.prev_gray[crop:]), 
            np.float32(curr_gray[crop:])
        )
        
        # Acumular el movimiento absoluto de la cámara
        self.cam_x -= shift[0]
        self.cam_y -= shift[1]

        # Posición absoluta de Mario en el mundo = posición en pantalla + posición de la cámara
        world_x = mx + self.cam_x
        world_y = my + self.cam_y

        vx = (world_x - self.world_prev[0]) / 16.0
        vy = (world_y - self.world_prev[1]) / 16.0
        true_vel = np.clip(np.array([vx, vy], dtype=np.float32), -2.0, 2.0)
        
        self.world_prev = (world_x, world_y)
        self.prev_gray = curr_gray
        # ----------------------------------------------------

        won = bool(info.get("flag_get"))
        died = done and not won
        reward = total / REWARD_SCALE
        if won:
            reward += WIN_BONUS
        if died:
            reward -= DEATH_PENALTY
        self.last_action = action
        return self._obs(self._base(frame, true_vel=true_vel)), reward, done, died, info


# ---------------------------------------------------------------
# RED (objetos + mapa sólido + GRU)
# ---------------------------------------------------------------
class ObjectReasoner(nn.Module):
    def __init__(self, n_actions, n_classes, K, obj_dim, base_dim, hidden_dim=128):
        super().__init__()
        self.n_actions, self.base_dim, self.hidden_dim = n_actions, base_dim, hidden_dim
        self.obj_dim = obj_dim

        self.register_buffer("cls_ids", torch.arange(n_classes).repeat_interleave(K))
        self.emb = nn.Embedding(n_classes, 16)
        self.tok = nn.Sequential(nn.Linear(5 + 16, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU())
        self.gate = nn.Linear(64, 1)

        conv_out = 16 * ((GRID_ROWS + 1) // 2) * ((GRID_COLS + 1) // 2)
        self.grid_enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(conv_out, 64), nn.ReLU())

        in_dim = 64 + 5 + 2 + 64 + n_actions   # objetos + Mario + vel + mapa + última acción
        self.feature_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.danger = nn.Linear(hidden_dim, n_actions)
        self.actor = nn.Sequential(nn.Linear(hidden_dim + n_actions, 128), nn.ReLU(),
                                   nn.Linear(128, n_actions))
        self.critic = nn.Linear(hidden_dim, 1)
        self.transition = nn.Sequential(nn.Linear(hidden_dim + n_actions, 128), nn.ReLU(),
                                        nn.Linear(128, obj_dim + 2))

    def forward(self, obs, h_state=None):
        B = obs.shape[0]
        if h_state is None:
            h_state = torch.zeros(B, self.hidden_dim, device=obs.device)

        cur = obs[:, :self.base_dim]
        last_a = obs[:, self.base_dim:]
        mario = cur[:, :5]
        tokens = cur[:, 5:self.obj_dim].reshape(B, -1, 5)   # [presente, dx, dy, ancho, alto]
        present = tokens[..., 0]
        vel = cur[:, self.obj_dim:self.obj_dim + 2]
        grid = cur[:, self.obj_dim + 2:].reshape(B, 1, GRID_ROWS, GRID_COLS)

        emb = self.emb(self.cls_ids).unsqueeze(0).expand(B, -1, -1)
        h_tok = self.tok(torch.cat([tokens, emb], dim=-1))
        relevance = torch.sigmoid(self.gate(h_tok)).squeeze(-1) * present
        pooled = (relevance.unsqueeze(-1) * h_tok).sum(dim=1)

        g = self.grid_enc(grid)
        x = self.feature_encoder(torch.cat([pooled, mario, vel, g, last_a], dim=1))
        new_h = self.gru(x, h_state)

        d_logits = self.danger(new_h)
        logits = self.actor(torch.cat([new_h, torch.sigmoid(d_logits).detach()], dim=1))
        value = self.critic(new_h).squeeze(-1)
        return logits, value, d_logits, new_h

    def predict_next(self, h, action_onehot):
        return self.transition(torch.cat([h, action_onehot], dim=1))


def to_t(x):
    return torch.as_tensor(x, device=DEVICE, dtype=torch.float32).unsqueeze(0)


def build(perception, env):
    return ObjectReasoner(env.n_actions, len(perception.other_ids), perception.K,
                          perception.dim_obj, env.base_dim).to(DEVICE)


def danger_labels(done, died, Hz):
    labels = torch.zeros(len(done))
    countdown = 1e9
    for t in reversed(range(len(done))):
        if died[t]:
            countdown = 0
        elif done[t]:
            countdown = 1e9
        else:
            countdown += 1
        labels[t] = 1.0 if countdown < Hz else 0.0
    return labels


# ---------------------------------------------------------------
# ENTRENAMIENTO (PPO recurrente por secuencias)
# ---------------------------------------------------------------
def train(total_steps=3_000_000, rollout=2048, seq_len=32, epochs=4, batch=256,
          lr=2.5e-4, clip=0.2, vf_coef=0.5, ent_coef_start=0.02,
          aux_coef=0.05, trans_coef=0.1):
    perception = RichPerception(find_weights())
    env = YoloOnlyEnv(perception)
    Td, n_act = env.trans_dim, env.n_actions
    model = build(perception, env)

    if os.path.exists(SAVE_PATH):
        try:
            model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
            print(f"Reanudando desde {SAVE_PATH}")
        except RuntimeError:
            os.replace(SAVE_PATH, SAVE_PATH.replace(".pt", "_antiguo.pt"))
            print("Pesos no compatibles: guardados como *_antiguo.pt, empiezo de cero.")

    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    obs = env.reset()
    h_state = torch.zeros(1, model.hidden_dim, device=DEVICE)
    ep_reward, ep_max_x = 0.0, 0
    ep_rewards, ep_xs = deque(maxlen=20), deque(maxlen=20)
    death_xs = deque(maxlen=50)
    deaths = episodes = 0
    steps = int(open(STEPS_PATH).read()) if os.path.exists(STEPS_PATH) else 0
    if steps:
        print(f"Pasos acumulados anteriores: {steps}")
    last_trans = last_danger = 0.0

    while steps < total_steps:
        F, A, LP, R, D, DIED, V, H_STATES = [], [], [], [], [], [], [], []

        for _ in range(rollout):
            f = to_t(obs)
            with torch.no_grad():
                logits, v, _, next_h = model(f, h_state)
                dist = Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)

            obs2, r, done, died, info = env.step(a.item())

            F.append(f.squeeze(0)); A.append(a.squeeze(0)); LP.append(lp.squeeze(0))
            R.append(r); D.append(float(done)); DIED.append(bool(died))
            V.append(v.squeeze(0)); H_STATES.append(h_state.squeeze(0))

            h_state = next_h
            ep_reward += r
            ep_max_x = max(ep_max_x, int(info.get("x_pos", 0)))
            obs = obs2

            if done:
                if died:
                    death_xs.append(int(info.get("x_pos", 0)))
                ep_rewards.append(ep_reward)
                ep_xs.append(ep_max_x)
                ep_reward, ep_max_x = 0.0, 0
                episodes += 1
                deaths += int(died)
                obs = env.reset()
                h_state = torch.zeros(1, model.hidden_dim, device=DEVICE)

        steps += rollout

        F, A, LP, V, H_STATES = map(torch.stack, (F, A, LP, V, H_STATES))
        R = torch.tensor(R, device=DEVICE, dtype=torch.float32)
        D = torch.tensor(D, device=DEVICE, dtype=torch.float32)
        danger = danger_labels(D.tolist(), DIED, DANGER_HORIZON).to(DEVICE)

        # objetivo de transición: cambio de objetos y velocidad (no del mapa), acotado
        F_next = torch.cat([F[1:], to_t(obs)], dim=0)
        target = ((F_next[:, :Td] - F[:, :Td]) * 5.0).clamp(-2.0, 2.0)
        valid = 1.0 - D
        reset = torch.zeros_like(D)
        reset[1:] = D[:-1]

        with torch.no_grad():
            _, last_v, _, _ = model(to_t(obs), h_state)
        adv, ret = compute_gae(R, V, D, last_v.squeeze(0))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        progress = steps / float(total_steps)
        ent_coef = max(0.01, ent_coef_start * (1.0 - progress))
        aux_scale = min(1.0, progress * 4.0)

        n_seq = rollout // seq_len
        F_s = F.view(n_seq, seq_len, -1)
        A_s = A.view(n_seq, seq_len)
        LP_s = LP.view(n_seq, seq_len)
        ADV_s = adv.view(n_seq, seq_len)
        RET_s = ret.view(n_seq, seq_len)
        DAN_s = danger.view(n_seq, seq_len)
        TGT_s = target.view(n_seq, seq_len, -1)
        VAL_s = valid.view(n_seq, seq_len)
        RST_s = reset.view(n_seq, seq_len)
        H0 = H_STATES.view(n_seq, seq_len, -1)[:, 0, :]

        seq_idx = np.arange(n_seq)
        seq_batch = max(1, batch // seq_len)

        for _ in range(epochs):
            np.random.shuffle(seq_idx)
            for i in range(0, n_seq, seq_batch):
                b = seq_idx[i:i + seq_batch]
                bF, bA, bLP, bADV = F_s[b], A_s[b], LP_s[b], ADV_s[b]
                bRET, bDAN, bTGT, bVAL, bRST = RET_s[b], DAN_s[b], TGT_s[b], VAL_s[b], RST_s[b]

                curr_h = H0[b]
                lo, va, dl, pr = [], [], [], []
                for t in range(seq_len):
                    curr_h = curr_h * (1.0 - bRST[:, t]).unsqueeze(-1)
                    logits_t, val_t, d_t, next_h = model(bF[:, t], curr_h)
                    a1h = Fn.one_hot(bA[:, t], n_act).float()
                    lo.append(logits_t); va.append(val_t); dl.append(d_t)
                    pr.append(model.predict_next(next_h, a1h))
                    curr_h = next_h

                logits_b = torch.stack(lo, 1)
                value_b = torch.stack(va, 1)
                d_logits_b = torch.stack(dl, 1)
                pred_b = torch.stack(pr, 1)

                dist = Categorical(logits=logits_b)
                ratio = (dist.log_prob(bA) - bLP).exp()
                s1 = ratio * bADV
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * bADV
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = Fn.smooth_l1_loss(value_b, bRET)
                entropy = dist.entropy().mean()

                d_taken = d_logits_b.gather(-1, bA.unsqueeze(-1)).squeeze(-1)
                danger_loss = Fn.binary_cross_entropy_with_logits(d_taken, bDAN)

                trans_err = (pred_b - bTGT).pow(2).mean(-1)
                trans_loss = (trans_err * bVAL).sum() / bVAL.sum().clamp(min=1.0)

                loss = (policy_loss + vf_coef * value_loss - ent_coef * entropy
                        + (aux_coef * danger_loss + trans_coef * trans_loss) * aux_scale)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                last_trans, last_danger = trans_loss.item(), danger_loss.item()

        mean_r = np.mean(ep_rewards) if ep_rewards else 0.0
        mean_x = np.mean(ep_xs) if ep_xs else 0.0
        where = ", ".join(f"x{x}-{x + 99}: {n}" for x, n in
                          Counter((x // 100) * 100 for x in death_xs).most_common(3))
        print(f"steps {steps:>9} | recompensa: {mean_r:6.2f} | x máx media: {mean_x:6.0f} | "
              f"muertes: {deaths}/{episodes} | err.trans: {last_trans:.3f} | "
              f"err.peligro: {last_danger:.3f}")
        print(f"            muere sobre todo en -> {where or 'aún sin datos'}")
        torch.save(model.state_dict(), SAVE_PATH)
        with open(STEPS_PATH, "w") as fh:
            fh.write(str(steps))


# ---------------------------------------------------------------
# VISOR
# ---------------------------------------------------------------
def draw_agent_view(obs, perception, width=280, height=160):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cx, cy = width // 2, height // 2
    sx, sy = 64, 60
    sw, sh = 16, 16  # scale for w, h (since token = w/32, and panel scale is 0.5)

    # Ejes
    cv2.line(panel, (0, cy), (width, cy), (70, 70, 70), 1)
    cv2.line(panel, (cx, 0), (cx, height), (70, 70, 70), 1)

    # Dibujar el mapa sólido que recibe PPO (suelo, bloques, tuberías)
    trans_dim = perception.dim_obj + 2
    grid_data = obs[trans_dim : trans_dim + GRID_ROWS * GRID_COLS].reshape(GRID_ROWS, GRID_COLS)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid_data[r, c] > 0.5:
                x_real = GRID_X0 + c * CELL
                y_real = GRID_Y0 + r * CELL
                x_pan = int(cx + x_real * 0.5)
                y_pan = int(cy + y_real * 0.5)
                cv2.rectangle(panel, (x_pan, y_pan), (x_pan + int(CELL * 0.5), y_pan + int(CELL * 0.5)), (100, 100, 100), -1)

    # Mario = centro (usando w y h reales que ve la red)
    mw, mh = obs[3], obs[4]
    m_w_px = max(1, int(mw * sw))
    m_h_px = max(1, int(mh * sh))
    m_x1, m_y1 = cx - m_w_px // 2, cy - m_h_px // 2
    m_x2, m_y2 = cx + m_w_px // 2, cy + m_h_px // 2
    cv2.rectangle(panel, (m_x1, m_y1), (m_x2, m_y2), (0, 0, 255), -1)

    # Extraer EXACTAMENTE los tokens que recibe la red (ignorando clases o nombres)
    tokens = obs[5:perception.dim_obj].reshape(-1, 5)
    for present, dx, dy, w, h in tokens:
        if present < 0.5:
            continue

        # dx/dy ya son exactamente los valores que recibe PPO
        px = int(cx + dx * sx)
        py = int(cy + dy * sy)

        # Tamaños reales que recibe PPO
        w_px = max(1, int(w * sw))
        h_px = max(1, int(h * sh))

        x1 = px - w_px // 2
        y1 = py - h_px // 2
        x2 = px + w_px // 2
        y2 = py + h_px // 2

        # Dibujar rectángulo en vez de un simple punto
        cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 255), 1)

    return panel


def play(greedy=False, scale=3, delay_ms=30):
    perception = RichPerception(find_weights())
    env = YoloOnlyEnv(perception)
    model = build(perception, env)
    if not os.path.exists(SAVE_PATH):
        print(f"No existe {SAVE_PATH}. Entrena primero con: python mario_ppo_yolo8.py")
        return
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()
    names = perception.model.names

    obs = env.reset()
    h_state = torch.zeros(1, model.hidden_dim, device=DEVICE)
    while True:
        with torch.no_grad():
            logits, _, d_logits, h_state = model(to_t(obs), h_state)
        a = logits.argmax(-1).item() if greedy else Categorical(logits=logits).sample().item()
        danger = torch.sigmoid(d_logits)[0].tolist()
        obs, _, done, _, _ = env.step(a)

        frame = env.env.unwrapped.screen.copy()
        xywh, cls, score = perception.detect(frame)
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), None,
                         fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for (cx, cy, w, h), c, s in zip(xywh, cls, score):
            x1, y1 = int((cx - w / 2) * scale), int((cy - h / 2) * scale)
            x2, y2 = int((cx + w / 2) * scale), int((cy + h / 2) * scale)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, f"{names[c]} {int(w)}x{int(h)}", (x1, max(y1 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        for i, (n, p) in enumerate(zip(ACTION_NAMES, danger)):
            mark = ">" if i == a else " "
            cv2.putText(img, f"{mark}{n}: {p:.2f}", (8, 20 + 16 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 255) if p > 0.5 else (255, 255, 255), 1)

        # Recuadro: EXACTAMENTE los dx/dy que recibe la red PPO
        g = draw_agent_view(obs, perception)
        gh, gw = g.shape[:2]
        img[8:8 + gh, img.shape[1] - gw - 8:img.shape[1] - 8] = g

        cv2.imshow("Mario (solo YOLO)", img)
        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break
        if done:
            obs = env.reset()
            h_state = torch.zeros(1, model.hidden_dim, device=DEVICE)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------
# EVALUACIÓN SIN VENTANA
# ---------------------------------------------------------------
def evaluate(n_episodes=20, max_steps=3000):
    perception = RichPerception(find_weights())
    env = YoloOnlyEnv(perception)
    model = build(perception, env)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    for greedy in (False, True):
        xs, wins = [], 0
        for _ in range(n_episodes):
            obs = env.reset()
            h = torch.zeros(1, model.hidden_dim, device=DEVICE)
            x_max, won = 0, False
            for _ in range(max_steps):
                with torch.no_grad():
                    logits, _, _, h = model(to_t(obs), h)
                a = (logits.argmax(-1).item() if greedy
                     else Categorical(logits=logits).sample().item())
                obs, _, done, died, info = env.step(a)
                x_max = max(x_max, int(info.get("x_pos", 0)))
                if done:
                    won = not died
                    break
            xs.append(x_max)
            wins += int(won)
        xs.sort()
        name = ("acción más probable (greedy)" if greedy
                else "sorteando acciones (como en el entrenamiento)")
        print(f"\n{name}:")
        print(f"  x máx de cada intento: {xs}")
        print(f"  mediana {xs[len(xs) // 2]} | media {sum(xs) / len(xs):.0f} | "
              f"llegan a la meta: {wins}/{n_episodes}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"play": play, "eval": evaluate}.get(mode, train)()
