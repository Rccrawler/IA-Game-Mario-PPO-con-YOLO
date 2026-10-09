# IA Game - Mario PPO con YOLO

![Mario IA Gameplay](gameplay.gif)

## Orden para que funcione

Sigue estos pasos para generar los datos, entrenar el sistema de visión y, por último, entrenar a la IA para jugar a Mario Bros.

**1. `python capture_frames.py`**
Genera las imágenes jugando en `dataset/images`.
*(Asegúrate de tener dentro de `templates/`, una carpeta por clase con tus recortes: ground, goomba, koopa, mario, coin, pipe, y block si quieres bloques aparte).*

**2. `python label_and_train.py preview`**
Revisas las cajas generadas en `dataset/preview`. Esto te permite asegurarte de que las plantillas detectan bien a los objetos.

**3. `python label_and_train.py label`**
Genera las etiquetas definitivas a partir de las coincidencias de las plantillas.

**4. `python label_and_train.py train`**
Entrena el modelo de detección visual YOLO.

**5. `python mario_ppo_yolo.py`**
Entrena el agente (la Inteligencia Artificial) usando PPO y la visión de YOLO para aprender a pasarse el nivel. Pulsa `Ctrl+C` en la consola para parar y guardar en cualquier momento.

**6. `python mario_ppo_yolo.py play`**
Lo ves jugar sin que siga entrenando.
