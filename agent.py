from collections import deque
from tqdm import tqdm
import random
import numpy as np
import tensorflow as tf
from networks import ValueFunction, Policy, EnvironmentModel, RewardFunction


class Agent:
    def __init__(self, env, gamma=0.95):
        self.gamma = gamma
        self.value_function = ValueFunction(env)
        self.environment_model = EnvironmentModel(env)
        self.reward_function = RewardFunction(env)
        self.policy = Policy(env)

        self.value_function.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        self.environment_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        self.reward_function.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        self.policy.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        self.policy_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    def train_environment_model(self, states, actions, next_states):
        """ Using a dataset of states and actions, train an environment model to predict
            the next state for a given start state and action
        """
        self.environment_model.fit(
            np.hstack([states, actions]), next_states, epochs=10, batch_size=32
        )

    def train_reward_function(self, states, actions, rewards):
        """ Using a dataset of states and actions, train a reward function to predict the reward
            (not the cumulative reward, just the one received at this timestep)
        """
        self.reward_function.fit(
            np.hstack([states, actions]), rewards, epochs=10, batch_size=32
        )

    def train_value_function(self, initial_states, trajectory_length=50):
        """ Starting from a set of initial states, use the environment model and the reward function to
            estimate the reward that the agent will accumulate in the immediate future and use that
            to train a value function.
        """
        states = initial_states
        values = tf.zeros(shape=(states.shape[0], 1), dtype=tf.float32)

        # Play out a trajectory of length T
        for t in tqdm(range(trajectory_length)):
            actions = self.policy(states)
            states = self.environment_model(np.hstack([states, actions]))
            rewards = self.reward_function(np.hstack([states, actions]))
            values = values + rewards * self.gamma ** t

        # Bottom out the recursion using the value function
        values = values + self.value_function(states) * self.gamma ** (t + 1)

        self.value_function.fit(
            tf.convert_to_tensor(initial_states), values, epochs=10, batch_size=32
        )

    def train_policy(self, states):
        """ Train the policy by maximising the value function over of a dataset of states.
        """
        dataset = tf.data.Dataset.from_tensor_slices(states.astype(np.float32))

        for i, S in tqdm(enumerate(dataset.batch(32))):
            with tf.GradientTape() as g:
                state_action = tf.concat([S, self.policy(S)], axis=1)
                reward = self.reward_function(state_action)
                value = reward + self.gamma * self.value_function(
                    self.environment_model(state_action)
                )
                loss = -tf.reduce_mean(value)

            policy_gradient = g.gradient(loss, self.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(
                zip(policy_gradient, self.policy.trainable_variables)
            )
