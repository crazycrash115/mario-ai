import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from gym_super_mario_bros.actions import RIGHT_ONLY  # Unused but can remove
from stable_baselines3.common.vec_env import DummyVecEnv

import gym
import cv2
import numpy as np
from gym.wrappers import ResizeObservation



# === Movement ===
CUSTOM_MOVEMENT = [
    ['NOOP'],           # No action
    ['right'],          # Walk right
    ['right', 'A'],     # Jump while walking right
    ['A'],              # Jump in place
    ['left'],           # Walk left
    ['left', 'A'],      # Jump while walking left
    ['B'],              # Run in place
    ['right', 'B'],     # Run right
    ['right', 'B', 'A'],# Run + Jump right 
    ['left', 'B'],      # Run left
    ['left', 'B', 'A'], # Run + Jump left
]

# === Action Repeat Wrapper ===
class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat=2):
        super(ActionRepeatWrapper, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

# === Wrapper to keep full-res frame ===
class VisualRenderWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_frame = None

    def observation(self, obs):
        self.last_frame = obs
        return obs

render_wrapper = None

def make_eval_env():
    global render_wrapper
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    env = JoypadSpace(env, CUSTOM_MOVEMENT)
    env = ActionRepeatWrapper(env, repeat=2)
    render_wrapper = VisualRenderWrapper(env)          # Save this
    env = ResizeObservation(render_wrapper, (84, 84))  # Then resize for AI
    return env


# === Run inference ===
if __name__ == '__main__':
    env = DummyVecEnv([make_eval_env])
    model = PPO.load("ppo_mario_latest", env)

    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        # Show full-resolution frame
        frame = render_wrapper.last_frame
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Full Resolution Mario", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if done:
            obs = env.reset()

    env.close()
    cv2.destroyAllWindows()