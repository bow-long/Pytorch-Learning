import torch

from key_lock_game import DQN, GameWindow, GameWorld, bundled_checkpoint, project_root


def main():
    best_path = project_root() / "best_generation.pt"
    checkpoint_path = best_path if best_path.exists() else bundled_checkpoint()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    model = DQN()
    model.load_state_dict(model_state)
    model.eval()

    saved_episode = checkpoint.get("episode", "bundled model")
    saved_score = checkpoint.get("score", "not recorded")
    world = GameWorld()
    window = GameWindow("Key-lock game - best generation")
    try:
        while not window.closed:
            delta_time = window.tick()
            with torch.no_grad():
                action = int(model(world.state()).argmax().item())
            _, _, done = world.step(action, delta_time)
            if done:
                world.reset()
            window.draw(
                world,
                [f"Checkpoint episode: {saved_episode}", f"Saved best score: {saved_score}"],
            )
    finally:
        window.close()


if __name__ == "__main__":
    main()
