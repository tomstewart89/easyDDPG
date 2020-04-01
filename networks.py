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
        self.dense1 = tf.keras.layers.Dense(400, activation="relu", input_shape=[state_dim])
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
        self.dense1 = tf.keras.layers.Dense(400, activation="relu", input_shape=[state_dim,])
        self.dense2 = tf.keras.layers.Dense(300, activation="relu")
        self.dense3 = tf.keras.layers.Dense(action_dim, activation="tanh")

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.dense3(x) * self.action_range


class FamiliarityFunction(tf.keras.Model):
    """ A VAE which encodes the states visited by the agent. We use this to explore by trying to find states that the function
    fails to encode well and also to represent the experience of the agent so that we can retrain the environment model without
    catastrophic forgetting or experience replay (in theory)
    """

    def __init__(self, env, latent_dim):
        super(FamiliarityFunction, self).__init__()

        state_dim = np.prod(env.observation_space.shape)
        action_dim = np.prod(env.action_space.shape)

        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(400, activation="relu"),
                tf.keras.layers.Dense(300, activation="relu"),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(300, activation="relu"),
                tf.keras.layers.Dense(400, activation="relu"),
                tf.keras.layers.Dense(state_dim + action_dim, activation=None),
            ]
        )

        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def log_normal_pdf(self, sample, mean, logvar):
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=1
        )

    @tf.function
    def compute_loss(self, state_action):
        mean, logvar = tf.split(self.inference_net(state_action), num_or_size_splits=2, axis=1)
        z = tf.random.normal(shape=mean.shape) * tf.exp(logvar * 0.5) + mean
        state_action_reconstructed = self.generative_net(z)

        logpx_z = -tf.keras.losses.mse(state_action_reconstructed, state_action)
        logpz = self.log_normal_pdf(z, 0.0, 0.0)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def train(self, state_action):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(state_action)
            # print(loss.numpy())

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
