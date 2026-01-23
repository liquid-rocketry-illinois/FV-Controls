from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
import random
import gymnasium as gym
from Maurice2.controls.silsim import SilSim
from Maurice2.controls.control_algorithm import Controls

class RocketEnv(gym.Env):
    
    metadata = {'render.modes': ['human', 'rgb_array'], "name": "RocketEnv-v0"}
    
    def __init__(
        self,
        sampling_rate: float = 40.0,
    ):
        self.sampling_rate = sampling_rate
        self.max_delta = np.deg2rad(8.0)  # Max aileron deflection angle
        self.controller = None
        self.silsim = None
        super().__init__()
    
    def set_controller(
        self,
        controller: Controls
    ):
        self.controller = controller
        
    def set_silsim(
        self,
        silsim: SilSim
    ):
        self.silsim = silsim
    
    def _get_state(self):
        state_list = self.silsim.rocketpy_state_to_xhat(self.silsim.states[-1])
        input = self.silsim.inputs[-1]
        state = {
            'position': state_list[0:3],
            'velocity': state_list[3:6],
            'orientation': state_list[6:10],
            'angular_velocity': state_list[10:13],
            'aileron_deflection': input[0],
        }
        return state

    def _get_obs(self):
        sensor_output = self.silsim.make_measurement_from_rocketpy(self.silsim.states[-1])
        obs = {
            "orientation": sensor_output[0:3],
            "angular_velocity": sensor_output[3:7],
            "aileron_deflection": self.silsim.inputs[-1][0],
        }
        
    
    def _get_reward(self):
        pass
    
    def reset(self):
        pass
    
    def step(self, action):
        pass
    
    def render(self, mode='human'):
        pass
    
    @property
    def observation_space(self):
        pass
    
    @property
    def action_space(self):
        action = Box(
            low=-self.max_delta, high = self.max_delta, shape=(1,), dtype=np.float32
        )
        return action