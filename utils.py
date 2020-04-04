import time
import tensorflow as tf
import numpy as np


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


@tf.function
def log_normal_pdf(sample, mean, logvar):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=1
    )
