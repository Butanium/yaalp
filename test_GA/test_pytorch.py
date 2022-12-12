import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pygame
import random
from torch.distributions.categorical import Categorical

PLAYER = 1
FOOD = 2
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class Board:
    def __init__(self, nb_food, size=(10, 10), max_steps=100, wall_penalty=1) -> None:
        self.wall_penalty = wall_penalty
        self.max_steps = max_steps
        self.steps = 0
        self.board = [[0 for i in range(size[1])] for j in range(size[0])]
        self.size = size
        self.board = np.array(self.board)
        self.player = self.spawn_player()
        self.board[self.player[0]][self.player[1]] = PLAYER
        self.food = []
        self.spawn_food(nb_food)
        self.score = 0
        self.failures = 0
        self.nb_food = nb_food
        # initializes the pygame window
        pygame.init()
        self.screen = pygame.display.set_mode((500, 500))

    def reward(self):
        return self.score

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        # if the player is on the edge of the board, he can't move
        if (
                self.player[0] == 0
                and action == LEFT
                or self.player[0] == self.size[0] - 1
                and action == RIGHT
                or self.player[1] == self.size[1] - 1
                and action == UP
                or self.player[1] == 0
                and action == DOWN
        ):
            self.failures += 1
            return self.board, self.reward() - self.wall_penalty, self.is_over(), {}
        self.board[self.player[0]][self.player[1]] = 0
        if action == UP:
            self.player[1] += 1
        elif action == DOWN:
            self.player[1] -= 1
        elif action == LEFT:
            self.player[0] -= 1
        elif action == RIGHT:
            self.player[0] += 1

        if self.board[self.player[0]][self.player[1]] == FOOD:
            self.score += 1
            self.food.remove(self.player)

        self.board[self.player[0]][self.player[1]] = PLAYER
        return self.board, self.reward(), self.is_over(), {}

    def spawn_player(self):
        return [
            random.randint(0, self.size[0] - 1),
            random.randint(0, self.size[1] - 1),
        ]

    def spawn_food(self, nb_food):
        for i in range(nb_food):
            rand = [
                random.randint(0, self.size[0] - 1),
                random.randint(0, self.size[1] - 1),
            ]
            while self.board[rand[0]][rand[1]] != 0:
                rand = [
                    random.randint(0, self.size[0] - 1),
                    random.randint(0, self.size[1] - 1),
                ]
            self.board[rand[0]][rand[1]] = FOOD
            self.food.append(rand)

    def reset(self, nb_food=None) -> np.ndarray:
        self.steps = 0
        if nb_food is None:
            nb_food = self.nb_food
        self.board = [[0 for i in range(self.size[1])] for j in range(self.size[0])]
        self.board = np.array(self.board)
        self.player = self.spawn_player()
        self.board[self.player[0]][self.player[1]] = PLAYER
        self.food = []
        self.spawn_food(nb_food)
        self.score = 0
        return self.board

    def __str__(self) -> str:
        # transposes the board to make it more readable
        # and print it in a pretty way
        transposed = np.transpose(self.board)
        string = "_" * (self.size[0] + 2) + "\n"
        for x in range(self.size[0]):
            string += "|"
            for y in range(self.size[1]):
                if transposed[x][y] == PLAYER:
                    string += "P "
                elif transposed[x][y] == FOOD:
                    string += "F "
                else:
                    string += "  "
            string += "|\n"
        return string + "_" * (self.size[0] + 2)

    # diplays the board in a pygame window
    def display(self):
        self.screen.fill((0, 0, 0))
        transp = self.board
        x_size = 500 / self.size[0]
        y_size = 500 / self.size[1]

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                if transp[x][self.size[1] - 1 - y] == PLAYER:
                    pygame.draw.rect(
                        self.screen,
                        (255, 255, 255),
                        (x * x_size, y * y_size, x_size, y_size),
                    )
                elif transp[x][self.size[1] - 1 - y] == FOOD:
                    pygame.draw.rect(
                        self.screen,
                        (255, 0, 0),
                        (x * x_size, y * y_size, x_size, y_size),
                    )
        pygame.display.update()

    def is_over(self):
        return self.steps >= self.max_steps or self.food == []


class CNNPlayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.linear = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Average pooling
        x = torch.mean(x, dim=(2, 3))
        return self.linear(x)

    def game_action(self, board):
        x = torch.as_tensor(board.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return torch.argmax(self.forward(x))

    def get_policy(self, obs):
        # Warning: obs has not always the same shape.
        logits = self(obs)
        return Categorical(logits=logits)

    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

    def compute_loss(self, obs, act, weights):
        logp = self.get_policy(obs).log_prob(act)
        return -(logp * weights).mean()


class HumanPlayer:
    def __init__(self) -> None:
        pass

    def game_action(self, board):
        # waits for the user to press a key
        # and returns the corresponding action
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            keys = pygame.key.get_pressed()  # checking pressed keys
            if keys[pygame.K_UP]:
                return UP
            elif keys[pygame.K_DOWN]:
                return DOWN
            elif keys[pygame.K_LEFT]:
                return LEFT
            elif keys[pygame.K_RIGHT]:
                return RIGHT


def show_simulation(player, delay=100, nb_food=5):
    board = Board(nb_food)
    board.display()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        action = player.game_action(board)
        print("Action:", action)
        board.step(action)
        board.display()
        pygame.time.delay(delay)


def gradient_train(model=None, lr=0.01, batch_size=100, nb_food=5, wall_penalty=10):
    if model is None:
        model = CNNPlayer()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # make some empty lists for logging.
    batch_obs = []  # for observations
    batch_acts = []  # for actions
    batch_weights = []  # for R(tau) weighting in policy gradient
    batch_rets = []  # for measuring episode returns # What is the return?
    batch_lens = []  # for measuring episode lengths
    env = Board(nb_food, wall_penalty=wall_penalty)
    # reset episode-specific variables
    obs = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over
    ep_rews = []  # list for rewards accrued throughout ep

    # collect experience by acting in the environment with current policy
    while True:
        # save obs
        batch_obs.append(torch.as_tensor(obs.copy(), dtype=torch.float32).unsqueeze(0))

        # act in the environment
        act = model.get_action(
            torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        obs, rew, done, _ = env.step(act)

        # save action, reward
        batch_acts.append(act)
        ep_rews.append(rew)

        if done:
            # if episode is over, record info about episode
            # Is the reward discounted?
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a|s) is R(tau)
            # Why do we use a constant vector here?
            batch_weights += [ep_ret] * ep_len

            # reset episode-specific variables
            obs, done, ep_rews = env.reset(), False, []

            # won't render again this epoch
            finished_rendering_this_epoch = True

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    # take a single policy gradient update step
    optimizer.zero_grad()
    batch_loss = model.compute_loss(
        obs=torch.stack(batch_obs),
        act=torch.as_tensor(batch_acts, dtype=torch.int32),
        weights=torch.as_tensor(batch_weights, dtype=torch.float32),
    )
    batch_loss.backward()
    optimizer.step()
    return batch_loss, batch_rets, batch_lens


if __name__ == "__main__":
    # show_simulation(HumanPlayer())
    model = CNNPlayer()
    for i in range(1000):
        loss, rewards, lengths = gradient_train(model)
        print(
            f"Epoch {i} : Loss: {loss}\n    mean reward: {np.mean(rewards)}\n   mean length: {np.mean(lengths)}"
        )
    show_simulation(model)
