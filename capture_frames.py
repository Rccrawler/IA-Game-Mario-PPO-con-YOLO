"""
Captura frames de Super Mario Bros para entrenar el detector de objetos.
Solo se guardan imágenes (píxeles). La posición x_pos del juego se usa
únicamente para dos cosas: detectar si Mario está atascado y repartir las
capturas por zonas distintas del nivel. No se guarda ni la usa el agente.

    pip install opencv-python gym-super-mario-bros
    python capture_frames.py
"""
import glob
import os
import random
from collections import defaultdict, deque

import cv2
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

OUT_DIR = "dataset/images"
N_FRAMES = 1000        # imágenes a guardar
LEVELS = ["1-1", "1-2", "1-3", "2-1", "3-1", "4-1", "5-1", "6-1"]
CLEAR_OLD = False      # False = continúa añadiendo a las que ya tienes
                       # True  = borra las anteriores y empieza de cero

SAVE_EVERY = 6         # intenta guardar cada N pasos
BIN_PX = 32            # una "zona" del nivel = 32 px de ancho
MAX_PER_BIN = 8        # máximo de imágenes por zona (enemigos y animaciones cambian)
MIN_DIFF = 6.0         # diferencia mínima con la última imagen guardada
STUCK_STEPS = 20       # pasos sin avanzar = atascado
GIVE_UP_STEPS = 400    # pasos sin batir su récord de x = cambia de nivel
MAX_STEPS = 1_500_000  # tope de seguridad
NO_SAVE_LIMIT = 150_000  # pasos seguidos sin guardar nada = zonas agotadas, termina

# Índices de SIMPLE_MOVEMENT
NOOP, RIGHT, RIGHT_A, RIGHT_B, RIGHT_A_B, A, LEFT = range(7)


def random_macro():
    """Secuencia de acciones mantenidas varios pasos (saltos largos incluidos)."""
    kind = random.choices(["run", "jump", "long", "walk"], [3, 3, 3, 1])[0]
    if kind == "run":
        return [RIGHT_B] * random.randint(8, 25)
    if kind == "jump":
        return [RIGHT_A_B] * random.randint(10, 25)
    if kind == "long":
        return [RIGHT_A_B] * random.randint(30, 50)
    return [RIGHT] * random.randint(5, 15)


def unstuck_macro(n):
    """Maniobras para saltar obstáculos cuando Mario no avanza."""
    if n <= 2:
        return [RIGHT_A_B] * 45                       # salto largo corriendo
    if n <= 4:
        return [LEFT] * 12 + [RIGHT_A_B] * 50         # retrocede y salta
    return random.choice([
        [A] * 30 + [RIGHT_A_B] * 40,                  # salto alto y luego largo
        [LEFT] * 25 + [RIGHT_B] * 20 + [RIGHT_A_B] * 50,  # coge carrerilla
    ])


def new_env(level):
    env = JoypadSpace(gym_super_mario_bros.make(f"SuperMarioBros-{level}-v0"),
                      SIMPLE_MOVEMENT)
    env.reset()
    return env


def step(env, action):
    out = env.step(action)
    obs, info = out[0], out[-1]
    done = out[2] if len(out) == 4 else (out[2] or out[3])
    return obs, done, info


def small_gray(img_rgb):
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (64, 60)).astype(np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if CLEAR_OLD:
        for f in glob.glob(f"{OUT_DIR}/frame_*.png"):
            os.remove(f)

    bins = defaultdict(int)       # (nivel, zona) -> imágenes guardadas
    existing = len(glob.glob(f"{OUT_DIR}/frame_*.png"))
    saved = 0 if CLEAR_OLD else existing
    if saved:
        print(f"Continuando: ya había {saved} imágenes")
    total_steps, episode, steps_since_save = 0, 0, 0
    last_small = None

    while (saved < N_FRAMES and total_steps < MAX_STEPS
           and steps_since_save < NO_SAVE_LIMIT):
        level = LEVELS[episode % len(LEVELS)]
        episode += 1
        env = new_env(level)
        queue = deque()
        best_x = stuck_ref = stuck_timer = stuck_count = since_best = t = 0

        while (saved < N_FRAMES and total_steps < MAX_STEPS
               and steps_since_save < NO_SAVE_LIMIT):
            if not queue:
                queue.extend(random_macro())
            obs, done, info = step(env, queue.popleft())
            t += 1
            total_steps += 1
            steps_since_save += 1
            x = int(info.get("x_pos", 0))

            # progreso y atascos
            if x > best_x:
                best_x, since_best = x, 0
            else:
                since_best += 1
            if x > stuck_ref + 1:
                stuck_ref, stuck_timer, stuck_count = x, 0, 0
            else:
                stuck_timer += 1
            if stuck_timer >= STUCK_STEPS:
                stuck_timer = 0
                stuck_count += 1
                queue.clear()
                queue.extend(unstuck_macro(stuck_count))

            # guardar solo si es una zona poco vista y la imagen es distinta
            if t % SAVE_EVERY == 0 and not done:
                key = (level, x // BIN_PX)
                if bins[key] < MAX_PER_BIN:
                    s = small_gray(obs)
                    if last_small is None or np.abs(s - last_small).mean() > MIN_DIFF:
                        cv2.imwrite(f"{OUT_DIR}/frame_{saved:05d}.png",
                                    cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
                        bins[key] += 1
                        saved += 1
                        last_small = s
                        steps_since_save = 0
                        if saved % 100 == 0:
                            print(f"{saved}/{N_FRAMES} guardadas (nivel {level}, x={x})")

            if done or since_best > GIVE_UP_STEPS:
                break
        env.close()

    print(f"Guardadas {saved} imágenes en {OUT_DIR} "
          f"({len(bins)} zonas distintas)")
    if saved < N_FRAMES:
        print("No se llegó al total: las zonas alcanzables se agotaron. "
              "Sube MAX_PER_BIN o añade más niveles a LEVELS.")


if __name__ == "__main__":
    main()
