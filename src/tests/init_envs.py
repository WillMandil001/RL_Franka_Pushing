from gymnasium.envs.registration import register

register( id="INB0104-v0", entry_point="src.tests:INB0104Env", max_episode_steps=1000, )