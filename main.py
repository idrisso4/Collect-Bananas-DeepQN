import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from agent import Agent
from evaluator import evaluate
from model import QNetwork
from trainer import train
from utils import read_config

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    config = read_config()

    device = torch.device(config["device"])

    env = UnityEnvironment(file_name=config["BANANA_ENVIRONMENT"])

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print("Number of agents:", len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print("Number of actions:", action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print("States look like:", state)
    state_size = len(state)
    print("States have length:", state_size)

    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    model_train = args.train
    model_eval = args.eval

    if model_train:

        scores = train(
            env=env,
            brain_name=brain_name,
            agent=agent,
            n_episodes=config["n_episodes"],
            max_t=config["max_t"],
            eps_start=config["eps_start"],
            eps_end=config["eps_end"],
            eps_decay=config["eps_decay"],
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel("Score")
        plt.xlabel("Episode")
        plt.show()

    if model_eval:

        model = QNetwork(state_size, action_size, 0).to(device)
        model.load_state_dict(torch.load("model.pth"))
        model.eval()

        evaluate(env, brain_name, model, device, n_episodes=100, max_t=1000)
