import argparse

from play_best_generation import main as play_best
from train_key_lock_game import main as train


def main():
    parser = argparse.ArgumentParser(description="Key-lock DQN game")
    parser.add_argument(
        "--mode",
        choices=("train", "play-best"),
        required=True,
        help="train the agent or play the best saved generation",
    )
    mode = parser.parse_args().mode
    if mode == "train":
        train()
    else:
        play_best()


if __name__ == "__main__":
    main()
