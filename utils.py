import time


def run_env(env, policy, render=False, max_steps=None):
    state = env.reset()
    done = False
    step = 0

    while not done:
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        yield state, action, reward, next_state, 0.0 if done else 1.0
        state = next_state

        if render:
            env.render()
            time.sleep(0.01)

        if max_steps is not None:
            if step == max_steps:
                break
