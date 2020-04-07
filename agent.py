from collections import deque
from tqdm import tqdm
import random
import numpy as np
import tensorflow as tf
from networks import ValueFunction, Policy, EnvironmentModel, RewardFunction, FamiliarityFunction


class Agent:
    def __init__(self, env, gamma=0.95, latent_dim=2):
        self.gamma = gamma
        self.value_function = ValueFunction(env)
        self.environment_model = EnvironmentModel(env)
        self.reward_function = RewardFunction(env)
        self.policy = Policy(env)
        self.familiarity_function = FamiliarityFunction(env, latent_dim)

        self.value_function.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        self.environment_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        self.reward_function.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        self.policy.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        self.policy_optimiser = tf.keras.optimizers.SGD(learning_rate=0.01)
        self.familiarity_optimiser = tf.keras.optimizers.Adam()

    def train_environment_model(self, states, actions, next_states):
        """ Using a dataset of states and actions, train an environment model to predict
            the next state for a given start state and action
        """
        self.environment_model.fit(
            np.hstack([states, actions]), next_states, epochs=3, batch_size=32
        )

    def train_reward_function(self, states, actions, rewards):
        """ Using a dataset of states and actions, train a reward function to predict the reward
            (not the cumulative reward, just the one received at this timestep)
        """
        self.reward_function.fit(np.hstack([states, actions]), rewards, epochs=3, batch_size=32)

    def train_value_function(
        self, initial_states, initial_rewards, next_states, trajectory_length=50
    ):
        """ Starting from a set of initial states, calculate the first step of the value using the information in the
            replay buffer and then forward predict the rest of the trajectory using the environment model and the reward function
            to calculate a target for the value function.
        """

        states = next_states
        values = initial_rewards

        # Play out a trajectory of length T
        for t in tqdm(range(1, trajectory_length + 1)):
            actions = self.policy(states)
            states = self.environment_model(np.hstack([states, actions]))
            rewards = self.reward_function(np.hstack([states, actions]))
            values = values + rewards * self.gamma ** t

        # Bottom out the recursion using the value function
        values = values + self.value_function(states) * self.gamma ** (trajectory_length + 1)

        self.value_function.fit(
            tf.convert_to_tensor(initial_states), values, epochs=3, batch_size=32
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
            self.policy_optimiser.apply_gradients(
                zip(policy_gradient, self.policy.trainable_variables)
            )

    def train_familiarity_function(self, states, actions, epochs=3):
        """ Train the familiarity function to effeciently encode states and actions so that
            we can easily spot new and interesting states while exploring.
        """
        dataset = (
            tf.data.Dataset.from_tensor_slices(np.hstack([states, actions]).astype(np.float32))
            .batch(32)
            .shuffle(10000)
        )

        for _ in tqdm(range(epochs)):
            for state_action in dataset:

                with tf.GradientTape() as tape:
                    loss = self.familiarity_function.loss(state_action)

                gradients = tape.gradient(loss, self.familiarity_function.trainable_variables)
                self.familiarity_optimiser.apply_gradients(
                    zip(gradients, self.familiarity_function.trainable_variables)
                )
