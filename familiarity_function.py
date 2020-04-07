import gym
from agent import Agent
from experience_replay import Experience
from introspect import plot_familiarity_latent_space, plot_familiarity_sample

if __name__ == "__main__":

    env = gym.make("Pendulum-v0")
    env.seed(0)

    replay_memory = Experience(filepath="pendulum_experience.pkl")
    agent = Agent(env, latent_dim=2)

    states, actions, rewards, next_states, _ = replay_memory.sample(len(replay_memory))

    agent.train_familiarity_function(states, actions, epochs=3)

    plot_familiarity_latent_space(agent, states, actions)
    plot_familiarity_sample(agent, 1000)
