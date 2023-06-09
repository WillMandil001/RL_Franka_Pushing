from gymnasium.envs.registration import register

register( id="gym_INB0104/INB0104-v0", entry_point="gym_INB0104.envs:INB0104Env" , max_episode_steps=1000, reward_threshold=100.0)
