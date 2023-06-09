import gymnasium as gym

import gym_INB0104

def main():
    env = gym.make("gym_INB0104/INB0104-v0")

    # print the observation space
    print("Observation space:", env.observation_space)

    # print the action space
    print("Action space:", env.action_space)

    # reset the environment
    # env.reset_model()
    data = env.reset()

    # render the environment
    env.render()
    
    for j in range(10):
        for i in range(100):
            env.step(env.action_space.sample())
            env.render()
        env.reset()

    # step the environment
    # env.step(env.action_space.sample())


if __name__ == "__main__":
    main()
