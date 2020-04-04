import tensorflow as tf
from utils import log_normal_pdf
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
    optimizer = tf.keras.optimizers.Adam()

    states, actions, rewards, next_states, _ = replay_memory.sample(len(replay_memory))
    dataset = tf.data.Dataset.from_tensor_slices(np.hstack([states, actions]).astype(np.float32))

    for epoch in range(3):
        for state_action in dataset.batch(32):

            with tf.GradientTape() as tape:
                loss = familiarity.loss(state_action)

            gradients = tape.gradient(loss, familiarity.trainable_variables)
            optimizer.apply_gradients(zip(gradients, familiarity.trainable_variables))
            print(loss.numpy())
