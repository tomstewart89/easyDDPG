import tensorflow as tf
import numpy as np


class EnvironmentModel(tf.keras.Model):
    """ Produces an estimate of the next state given the current state and an action
    """

    def __init__(self, env):
        super(EnvironmentModel, self).__init__()
        self.state_dim, self.action_dim = (
            np.prod(env.observation_space.shape),
            np.prod(env.action_space.shape),
        )
        self.dense1 = tf.keras.layers.Dense(
            400, activation="relu", input_shape=[self.state_dim + self.action_dim,]
        )
        self.dense2 = tf.keras.layers.Dense(300, activation="relu")
        self.dense3 = tf.keras.layers.Dense(self.state_dim, activation=None)

    def call(self, state_action):
        x = self.dense1(state_action)
        x = self.dense2(x)
        x = self.dense3(x)
        return state_action[:, : self.state_dim] + x


class RewardFunction(tf.keras.Model):
    """ Produces an estimate the reward accrued in one timestep by making an action in a given state
    """

    def __init__(self, env):
        super(RewardFunction, self).__init__()
        state_dim, action_dim = (
            np.prod(env.observation_space.shape),
            np.prod(env.action_space.shape),
        )
        self.dense1 = tf.keras.layers.Dense(
            400, activation="relu", input_shape=[state_dim + action_dim,]
        )
        self.dense2 = tf.keras.layers.Dense(300, activation="relu")
        self.dense3 = tf.keras.layers.Dense(1, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.dense3(x)


class ValueFunction(tf.keras.Model):
    """ Maps a state to an estimate of the reward we expect to accumulate starting from that state between now and the end of the episode
    """

    def __init__(self, env):
        super(ValueFunction, self).__init__()
        state_dim, action_dim = (
            np.prod(env.observation_space.shape),
            np.prod(env.action_space.shape),
        )
        self.dense1 = tf.keras.layers.Dense(
            400, activation="relu", input_shape=[state_dim]
        )
        self.dense2 = tf.keras.layers.Dense(300, activation="relu")
        self.dense3 = tf.keras.layers.Dense(1, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.dense3(x)


class Policy(tf.keras.Model):
    """ Maps a state to an action
    """

    def __init__(self, env):
        super(Policy, self).__init__()
        state_dim, action_dim = (
            np.prod(env.observation_space.shape),
            np.prod(env.action_space.shape),
        )
        self.action_range = env.action_space.high - env.action_space.low
        self.dense1 = tf.keras.layers.Dense(
            400, activation="relu", input_shape=[state_dim,]
        )
        self.dense2 = tf.keras.layers.Dense(300, activation="relu")
        self.dense3 = tf.keras.layers.Dense(action_dim, activation="tanh")

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.dense3(x) * self.action_range