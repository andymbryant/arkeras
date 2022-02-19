import time
import numpy as np
import gym
from gym import spaces

import pygame
import random
from pygame.surfarray import array3d
from pygame import display
from src.lib.vars import BLACK, WHITE, GREEN
from src.games.snake.vars import (
    ACTION_TO_DIRECTION_MAP,
    DEFAULT_DIRECTION,
    DEFAULT_SNAKE_LENGTH,
    DIRECTION_TO_OPPOSITE_MAP,
    DOWN,
    GAME_UNIT_SIZE,
    LEFT,
    REWARD_HIGH,
    REWARD_LOW,
    REWARD_NEUTRAL,
    RIGHT,
    UP,
)

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.frame_size_x = 200
        self.frame_size_y = 200
        self.game_window = pygame.display.set_mode(
            (self.frame_size_x, self.frame_size_y))
        self.name = 'snake'
        self.reset()

    def step(self, action):
        self.steps += 1
        reward, done = self.is_game_over()
        if not done:
            self.change_direction(action)
            self.make_move()
            reward = self.food_handler()
        self.redraw()
        img = self.get_image_array_from_game()
        info = {"score": self.score}
        return img, reward, done, info

    def change_direction(self, action):
        new_direction = ACTION_TO_DIRECTION_MAP.get(action)
        # If new_direction is opposite of current direction, new_direction is illegal
        if DIRECTION_TO_OPPOSITE_MAP.get(new_direction) != self.direction:
            self.direction = new_direction

    def make_move(self):
        if self.direction == UP:
            self.snake_pos[1] -= GAME_UNIT_SIZE
        elif self.direction == DOWN:
            self.snake_pos[1] += GAME_UNIT_SIZE
        elif self.direction == LEFT:
            self.snake_pos[0] -= GAME_UNIT_SIZE
        elif self.direction == RIGHT:
            self.snake_pos[0] += GAME_UNIT_SIZE
        self.snake_body.insert(0, list(self.snake_pos))

    def did_eat_food(self):
        return self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]

    def spawn_food(self):
        return [random.randrange(1, (self.frame_size_x//GAME_UNIT_SIZE)) * GAME_UNIT_SIZE, random.randrange(1, (self.frame_size_y//GAME_UNIT_SIZE)) * GAME_UNIT_SIZE]

    def food_handler(self):
        if self.did_eat_food():
            self.score += 1
            reward = REWARD_HIGH
            self.food_pos = self.spawn_food()
        else:
            self.snake_body.pop()
            reward = REWARD_NEUTRAL
        return reward

    def redraw(self):
        self.game_window.fill(BLACK)
        for x, y in self.snake_body:
            pygame.draw.rect(self.game_window, GREEN,
                             pygame.Rect(x, y, GAME_UNIT_SIZE, GAME_UNIT_SIZE))

        pygame.draw.rect(self.game_window, WHITE, pygame.Rect(
            self.food_pos[0], self.food_pos[1], GAME_UNIT_SIZE, GAME_UNIT_SIZE))

    def get_image_array_from_game(self):
        img = array3d(display.get_surface())
        img = np.swapaxes(img, 0, 1)
        return img

    def is_game_over(self):
        if self.score == 100:
            return REWARD_HIGH, True
        if self.steps == 1000:
            return REWARD_LOW, True
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x - GAME_UNIT_SIZE:
            return REWARD_LOW, True
        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y - GAME_UNIT_SIZE:
            return REWARD_LOW, True
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                return REWARD_LOW, True
        return REWARD_NEUTRAL, False

    def reset(self):
        self.game_window.fill(BLACK)
        self.snake_pos = [100, 50]
        self.snake_body = [[self.snake_pos[0] - (GAME_UNIT_SIZE * i), self.snake_pos[1]] for i, _ in enumerate(range(DEFAULT_SNAKE_LENGTH))]
        self.food_pos = self.spawn_food()

        self.direction = DEFAULT_DIRECTION
        self.score = 0
        self.steps = 0
        return self.get_image_array_from_game()

    def render(self, mode='human'):
        if mode == "human":
            display.update()
            time.sleep(0.01)

    def close(self):
        pass
