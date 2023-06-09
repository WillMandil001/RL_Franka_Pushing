import gymnasium as gym

import gym_INB0104

def main():
    env = gym.make("gym_INB0104/INB0104-v0")

    # print the observation space
    print("Observation space:", env.observation_space)

    # print the action space
    print("Action space:", env.action_space)

    # reset the environment
    env.reset_model()

    # render the environment
    env.render()

    # step the environment
    env.step(env.action_space.sample())


if __name__ == "__main__":
    main()
