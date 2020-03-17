import gym
from tqdm import tqdm
from agent import Agent
from experience_replay import Experience
from utils import run_env
from introspect import test_forward_prediction

if __name__ == "__main__":

    env = gym.make("Pendulum-v0")
    env.seed(0)

    replay_memory = Experience(1e5)
    agent = Agent(env)

    test_forward_prediction(agent, env)

    # Gather up a heap of experience
    for _ in tqdm(range(500)):
        for transition in run_env(env, lambda s: env.action_space.sample()):
            replay_memory.store(*transition)

    states, actions, rewards, next_states, _ = replay_memory.sample(len(replay_memory))

    # Now have the agent learn about its environment
    agent.train_environment_model(states, actions, next_states)
    agent.train_reward_function(states, actions, rewards)

    # Now have the agent learn its value function using its own internal models
    for _ in range(2):
        agent.train_value_function(states)
        agent.train_policy(states)
