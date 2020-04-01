import gym
from agent import Agent
from experience_replay import Experience
from introspect import plot_value_function

if __name__ == "__main__":

    env = gym.make("Pendulum-v0")
    env.seed(0)

    replay_memory = Experience(filepath="pendulum_experience.pkl")
    agent = Agent(env)

    states, actions, rewards, next_states, _ = replay_memory.sample(len(replay_memory))

    # Now have the agent learn about its environment
    agent.train_environment_model(states, actions, next_states)
    agent.train_reward_function(states, actions, rewards)

    agent.train_value_function(states, rewards, next_states, trajectory_length=0)

    plot_value_function(agent, states)

    # for t in reversed(range(0, 50, 10)):
    #     agent.train_value_function(states, rewards, next_states, trajectory_length=t)

    # plot_value_function(agent, states)
