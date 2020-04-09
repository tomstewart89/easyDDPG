import gym
from agent import Agent
from experience_replay import Experience
from introspect import plot_familiarity_latent_space, plot_familiarity_sample

if __name__ == "__main__":

    env = gym.make("Pendulum-v0")
    env.seed(0)

    on_policy_replay_memory = Experience(filepath="on_policy_experience.pkl")
    exploratory_replay_memory = Experience(filepath="exploratory_experience.pkl")

    agent = Agent(env, latent_dim=3)

    on_pol_states, _, _, _, _ = on_policy_replay_memory.sample(len(on_policy_replay_memory))
    expl_states, _, _, _, _ = exploratory_replay_memory.sample(len(exploratory_replay_memory))

    agent.train_familiarity_function(on_pol_states, epochs=5)

    plot_familiarity_latent_space(agent, on_pol_states)
    plot_familiarity_sample(agent, 1000)
