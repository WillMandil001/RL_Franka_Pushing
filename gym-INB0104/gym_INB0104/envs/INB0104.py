import numpy as np
import os

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    }


class INB0104Env(MujocoEnv, utils.EzPickle):
    metadata = { 
        "render_modes": [ 
            "human",
            "rgb_array", 
            "depth_array"
        ], 
        "render_fps": 100
    }
    
    def __init__(self, render_mode=None, use_distance=False, **kwargs):
        utils.EzPickle.__init__(self, use_distance, **kwargs)
        self.use_distance = use_distance
        observation_space = Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)
        cdir = os.getcwd()
        env_dir = os.path.join(cdir, "environments/INB0104/Robot_C.xml")
        MujocoEnv.__init__(self, env_dir, 5, observation_space=observation_space, default_camera_config=DEFAULT_CAMERA_CONFIG, camera_id=0, **kwargs,)
        self.render_mode = render_mode

    def step(self, a):
        vec = self.get_body_com("left_finger") - self.get_body_com("target_object")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        return (
            ob, 
            reward_dist, 
            False, 
            False, 
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
            )

    def reset_model(self):
        # set up random initial state for the robot - but keep the fingers in place
        qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04, 0.655, 0.515, 1.0, 0, 0, 0, 1])
        qpos[0] += self.np_random.uniform(low=-1, high=1)
        qpos[1] += self.np_random.uniform(low=-1, high=1)
        qpos[2] += self.np_random.uniform(low=-1, high=1)
        qpos[3] += self.np_random.uniform(low=-1, high=1)
        qpos[4] += self.np_random.uniform(low=-1, high=1)
        qpos[5] += self.np_random.uniform(low=-1, high=1)
        qpos[6] += self.np_random.uniform(low=-1, high=1)
        # qpos = ( self.np_random.uniform(low=-0.4, high=0.4, size=self.model.nq) + self.init_qpos)

        # create random x and y position for the target object, but make sure it is within a 1 meter circle -- this bit maybe useful later - not for now though
        while True:
            self.goal = self.np_random.uniform(low=-0.25, high=0.25, size=2)
            if np.linalg.norm(self.goal) < 1.0:
                break
        qpos[9:11] += self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[9:11] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        
        # theta = self.data.qpos.flat[:2]
        # return np.concatenate([np.cos(theta), np.sin(theta),
        #                        self.data.qpos.flat[2:], self.data.qvel.flat[:2],
        #                        self.get_body_com("left_finger") - self.get_body_com("target_object")])
        
        position = self.data.qpos[0:9].flat.copy()
        velocity = self.data.qvel[0:9].flat.copy()
        if self.use_distance:
            return np.concatenate([position, velocity, self.get_body_com("left_finger") - self.get_body_com("target_object")])
        else:
            return np.concatenate([position, velocity])


