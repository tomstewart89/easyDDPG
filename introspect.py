import numpy as np
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
                        lambda s: agent.policy(s.reshape(3, 1)),
                        max_steps=episode_length,
                        render=False,
                    )
                ]
            )
        )

    color = cm.jet(np.linspace(0.75, 1, 200))

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
