import gym
from tqdm import tqdm
from agent import Agent
from experience_replay import Experience
from utils import run_env
from introspect import *

if __name__ == "__main__":

    env = gym.make("Pendulum-v0")
    env.seed(0)

    replay_memory = Experience(1e5)
    agent = Agent(env)

    # Gather up a heap of experience
    for _ in tqdm(range(500)):
        for transition in run_env(
            env, lambda s: agent.policy(s.reshape(-1, 3)).numpy().reshape(1,)
        ):
            replay_memory.store(*transition)

    states, actions, rewards, next_states, _ = replay_memory.sample(len(replay_memory))

    # Now have the agent learn about its environment
    agent.train_environment_model(states, actions, next_states)
    agent.train_reward_function(states, actions, rewards)
    agent.train_familiarity_function(states)

    test_forward_prediction(agent, env)

    # Now have the agent learn its value function using its own internal models
    for _ in range(5):
        agent.train_value_function(states, rewards, next_states)
        agent.train_policy(states)

    plot_value_function(agent, states)
    plot_policy(agent, states)

    plot_trajectories(agent, env)
