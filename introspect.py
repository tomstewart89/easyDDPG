import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from utils import run_env


def test_forward_prediction(agent, env):
    transitions = [trans for trans in run_env(env, lambda s: env.action_space.sample())]
    states, actions, rewards, _, _ = map(list, zip(*transitions))

    predicted_states, predicted_rewards = [states[0]], []

    for action in actions:
        state_action = np.hstack([predicted_states[-1], action]).reshape(1, 4)
        predicted_states.append(agent.environment_model(state_action).numpy()[0])
        predicted_rewards.append(agent.reward_function(state_action).numpy()[0])

    model_prediction_error = np.subtract(predicted_states[:-1], states)
    reward_prediction_error = np.array(predicted_rewards).squeeze() - rewards

    for index, data in enumerate(
        [
            states,
            predicted_states,
            model_prediction_error,
            rewards,
            predicted_rewards,
            reward_prediction_error,
        ]
    ):
        plt.subplot(2, 3, index + 1)
        plt.plot(data)

    plt.show()


def plot_trajectories(agent, env, episode_length=100):
    trajectories = []

    for i in tqdm(range(20)):
        trajectories.append(
            np.array(
                [
                    transition[0].squeeze()
                    for transition in run_env(
                        env,
                        lambda s: agent.policy(s.reshape(-1, 3)),
                        max_steps=episode_length,
                        render=True,
                    )
                ]
            )
        )

    color = cm.jet(np.linspace(0, 1, trajectories[0].shape[0]))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    for trajectory in trajectories:
        ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c=color)

    plt.show()


def plot_value_function(agent, states):
    values = agent.value_function(states).numpy()
    color = cm.jet((values - values.min()) / (values - values.min()).max())

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(states[:, 0], states[:, 1], states[:, 2], c=color.squeeze())

    plt.show()


def plot_policy(agent, states):
    actions = agent.policy(states).numpy()[:, 0]
    color = cm.jet((actions - actions.min()) / (actions - actions.min()).max())

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(states[:, 0], states[:, 1], states[:, 2], c=color.squeeze())

    plt.show()


def plot_familiarity_latent_space(agent, states, actions):
    mean, logvar = tf.split(
        agent.familiarity_function.inference_net(np.hstack([states, actions]).astype(np.float32)),
        num_or_size_splits=2,
        axis=1,
    )

    z = tf.random.normal(shape=mean.shape) * tf.exp(logvar * 0.5) + mean

    for i, z_i in enumerate(z.numpy().T):
        plt.subplot(agent.familiarity_function.latent_dim, 1, i + 1)
        plt.hist(z_i, bins=50, range=(-3, 3))

    plt.show()


def plot_familiarity_sample(agent, n_samples):
    samples = agent.familiarity_function.sample(n_samples)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])

    plt.show()
