import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper
from wrappers import FrameStack
import cv2

import gym_INB0104

def main():
    env = gym.make("gym_INB0104/INB0104-v0", render_mode="rgb_array")
    env = PixelObservationWrapper(env, pixels_only=False)
    env = FrameStack(env, num_frames=3)
    
    # # print the observation space
    print("Observation space:", env.observation_space)
    # # print the action space
    print("Action space:", env.action_space)

    # reset the environment
    obs, info = env.reset()
    
    # render the environment
    while True:
        for i in range(100):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())


        env.reset()


if __name__ == "__main__":
    main()
