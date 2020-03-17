def run_env(env, policy):
    state = env.reset()
    done = False

    while not done:
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        yield state, action, reward, next_state, 0.0 if done else 1.0
        state = next_state
