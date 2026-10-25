import math
import random
import sys
from pathlib import Path

import pygame
import torch
from torch import nn
from torch.nn import functional as F


WIDTH = 1280
HEIGHT = 720
PLAYER_RADIUS = 100
HIT_TOLERANCE = 0.2
MAX_EPISODE_STEPS = 1800
N_OBSERVATIONS = 5
N_ACTIONS = 2


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(N_OBSERVATIONS, 256)
        self.layer4 = nn.Linear(256, N_ACTIONS)

    def forward(self, state):
        return self.layer4(F.relu(self.layer1(state)))


class GameWorld:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player_angle = random.random() * math.tau
        self.target_angle = random.random() * math.tau
        self.direction = random.choice((-1, 1))
        self.speed = 5.0
        self.score = 0
        self.steps = 0

    def state(self):
        distance = self.distance()
        return torch.tensor(
            (
                self.player_angle / math.tau,
                self.target_angle / math.tau,
                distance / math.pi,
                1.0 if self.direction == 1 else 0.0,
                self.speed / 10.0,
            ),
            dtype=torch.float32,
        )

    def distance(self):
        difference = abs(self.target_angle - self.player_angle)
        return min(difference, math.tau - difference)

    def step(self, action, delta_time):
        self.steps += 1
        self.player_angle = (
            self.player_angle + self.direction * self.speed * delta_time
        ) % math.tau

        reward = 0.0
        done = False
        if action == 1:
            distance = self.distance()
            if distance <= HIT_TOLERANCE:
                reward = 1.0 + 0.25 * (1.0 - distance / HIT_TOLERANCE)
                self.score += 1
                self.direction *= -1
                self.speed = min(self.speed * 1.1, 20.0)
                self.target_angle = random.random() * math.tau
            else:
                reward = -1.0
                done = True

        if self.steps >= MAX_EPISODE_STEPS:
            reward = min(reward, -0.25)
            done = True
        return self.state(), reward, done


def project_root():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundled_checkpoint():
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "tensor.pt"
    return project_root() / "tensor.pt"


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model = DQN()
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def save_checkpoint(path, checkpoint):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


class GameWindow:
    def __init__(self, title):
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22)
        self.closed = False

    def tick(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.closed = True
        return min(self.clock.tick(60) / 1000.0, 0.05)

    def draw(self, world, details):
        center = (WIDTH // 2, HEIGHT // 2)
        self.screen.fill((0, 0, 0))
        pygame.draw.circle(self.screen, (60, 60, 60), center, PLAYER_RADIUS, 1)
        player_position = self._position(world.player_angle)
        target_position = self._position(world.target_angle)
        pygame.draw.circle(self.screen, (0, 100, 0), player_position, 3)
        pygame.draw.circle(self.screen, (0, 100, 255), player_position, 10)
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            pygame.Rect(target_position[0] - 7, target_position[1] - 7, 14, 14),
        )

        lines = [
            f"Score: {world.score}",
            f"Distance: {world.distance():.3f} rad",
            f"Speed: {world.speed:.1f}",
            *details,
            "ESC: quit",
        ]
        for index, text in enumerate(lines):
            surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (24, 24 + index * 32))
        pygame.display.flip()

    @staticmethod
    def _position(angle):
        return (
            WIDTH // 2 + PLAYER_RADIUS * math.sin(angle),
            HEIGHT // 2 - PLAYER_RADIUS * math.cos(angle),
        )

    def close(self):
        pygame.quit()
