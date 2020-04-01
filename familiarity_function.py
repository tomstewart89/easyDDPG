import tensorflow as tf
from networks import FamiliarityFunction
from experience_replay import Experience
import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm

if __name__ == "__main__":

    env = gym.make("Pendulum-v0")
    env.seed(0)

    replay_memory = Experience(filepath="pendulum_experience.pkl")
    familiarity = FamiliarityFunction(env, 2)

    states, actions, rewards, next_states, _ = replay_memory.sample(len(replay_memory))

    dataset = tf.data.Dataset.from_tensor_slices(np.hstack([states, actions]).astype(np.float32))

    for state_action in tqdm(dataset.batch(32)):
        familiarity.train(state_action)

    r = 0.0
