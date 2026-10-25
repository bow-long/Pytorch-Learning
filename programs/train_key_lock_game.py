import random
from collections import deque, namedtuple

import torch
from torch import nn
from torch.nn import functional as F

from key_lock_game import (
    DQN,
    GameWindow,
    GameWorld,
    bundled_checkpoint,
    project_root,
    save_checkpoint,
)


BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
MEMORY_CAPACITY = 10_000
TARGET_UPDATE_STEPS = 500
SAVE_EPISODES = 10
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


def optimize(model, target_model, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return False

    batch = Transition(*zip(*random.sample(memory, BATCH_SIZE)))
    states = torch.stack(batch.state)
    actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1)
    next_states = torch.stack(batch.next_state)
    rewards = torch.tensor(batch.reward, dtype=torch.float32)
    dones = torch.tensor(batch.done, dtype=torch.float32)

    current_q = model(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q = target_model(next_states).max(1).values
        target_q = rewards + GAMMA * next_q * (1.0 - dones)

    loss = F.smooth_l1_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return True


def restore_training(model, target_model, optimizer, path):
    if not path.exists():
        initial_checkpoint = torch.load(
            bundled_checkpoint(), map_location="cpu", weights_only=True
        )
        model.load_state_dict(initial_checkpoint)
        target_model.load_state_dict(model.state_dict())
        return 0, EPSILON_START, -1

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    target_model.load_state_dict(checkpoint["target_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        int(checkpoint["episode"]),
        float(checkpoint["epsilon"]),
        int(checkpoint["best_score"]),
    )


def main():
    root = project_root()
    training_path = root / "training_checkpoint.pt"
    best_path = root / "best_generation.pt"

    model = DQN()
    target_model = DQN()
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    episode, epsilon, best_score = restore_training(
        model, target_model, optimizer, training_path
    )
    memory = deque(maxlen=MEMORY_CAPACITY)
    world = GameWorld()
    window = GameWindow("Key-lock game - training")
    steps_since_target_update = 0

    try:
        while not window.closed:
            delta_time = window.tick()
            state = world.state()
            if random.random() < epsilon:
                action = random.randrange(2)
            else:
                with torch.no_grad():
                    action = int(model(state).argmax().item())

            next_state, reward, done = world.step(action, delta_time)
            memory.append(Transition(state, action, next_state, reward, done))
            if optimize(model, target_model, optimizer, memory):
                steps_since_target_update += 1
            if steps_since_target_update >= TARGET_UPDATE_STEPS:
                target_model.load_state_dict(model.state_dict())
                steps_since_target_update = 0

            if done:
                episode += 1
                epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
                if world.score > best_score:
                    best_score = world.score
                    save_checkpoint(
                        best_path,
                        {
                            "model_state_dict": model.state_dict(),
                            "episode": episode,
                            "score": best_score,
                        },
                    )
                if episode % SAVE_EPISODES == 0:
                    save_checkpoint(
                        training_path,
                        {
                            "model_state_dict": model.state_dict(),
                            "target_state_dict": target_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "episode": episode,
                            "epsilon": epsilon,
                            "best_score": best_score,
                        },
                    )
                world.reset()

            window.draw(
                world,
                [
                    f"Episode: {episode}",
                    f"Best score: {best_score}",
                    f"Exploration: {epsilon:.3f}",
                ],
            )
    finally:
        save_checkpoint(
            training_path,
            {
                "model_state_dict": model.state_dict(),
                "target_state_dict": target_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "episode": episode,
                "epsilon": epsilon,
                "best_score": best_score,
            },
        )
        window.close()


if __name__ == "__main__":
    main()
