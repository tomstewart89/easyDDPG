import tensorflow as tf
import numpy as np
from utils import log_normal_pdf


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

    @tf.function
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

    @tf.function
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
        self.dense1 = tf.keras.layers.Dense(400, activation="relu", input_shape=[state_dim])
        self.dense2 = tf.keras.layers.Dense(300, activation="relu")
        self.dense3 = tf.keras.layers.Dense(1, activation=None)

    @tf.function
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
        self.dense1 = tf.keras.layers.Dense(400, activation="relu", input_shape=[state_dim,])
        self.dense2 = tf.keras.layers.Dense(300, activation="relu")
        self.dense3 = tf.keras.layers.Dense(action_dim, activation="tanh")

    @tf.function
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.dense3(x) * self.action_range


class FamiliarityFunction(tf.keras.Model):
    """ A VAE which encodes the states visited by the agent. We use this to explore by trying to find states that the function
    fails to encode well and also to represent the experience of the agent so that we can retrain the environment model without
    catastrophic forgetting or experience replay (in theory)
    """

    def __init__(self, env, latent_dim=2):
        super(FamiliarityFunction, self).__init__()

        self.latent_dim = latent_dim
        state_dim = np.prod(env.observation_space.shape)

        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(400, activation="relu"),
                tf.keras.layers.Dense(300, activation="relu"),
                tf.keras.layers.Dense(self.latent_dim * 2),
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(300, activation="relu"),
                tf.keras.layers.Dense(400, activation="relu"),
                tf.keras.layers.Dense(state_dim, activation=None),
            ]
        )

    @tf.function
    def call(self, states):
        mean, logvar = tf.split(self.inference_net(states), num_or_size_splits=2, axis=1)
        z = tf.random.normal(shape=mean.shape) * tf.exp(logvar * 0.5) + mean
        return self.generative_net(z)

    @tf.function
    def sample(self, n):
        z = tf.random.normal(shape=(n, self.latent_dim))
        return self.generative_net(z)

    @tf.function
    def loss(self, states, decoder_logvar=-6.0):
        mean, logvar = tf.split(self.inference_net(states), num_or_size_splits=2, axis=1)
        z = tf.random.normal(shape=mean.shape) * tf.exp(logvar * 0.5) + mean

        logpx_z = log_normal_pdf(states, self.generative_net(z), decoder_logvar)
        logpz = log_normal_pdf(z, 0.0, 0.0)
        logqz_x = log_normal_pdf(z, mean, logvar)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
