import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import numpy as np
import gym
import re
from gym.wrappers import ResizeObservation #resizes the screen for less ram use 

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

# === Reward Wrapper ===
class MarioRewardWrapper(gym.Wrapper):
    def __init__(self, env, max_frames=3000):
        super(MarioRewardWrapper, self).__init__(env)
        self.prev_x = 0
        self.prev_score = 0
        self.no_move_counter = 0
        self.frame_count = 0
        self.max_frames = max_frames
        self.max_x = 0


    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_x = 0
        self.prev_score = 0
        self.no_move_counter = 0
        self.frame_count = 0
        self.max_x = 0
        self.checkpoint_hit = False
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frame_count += 1

        new_x = info.get("x_pos", 0)
        new_score = info.get("score", 0)
        flag_get = info.get("flag_get", False)

        shaped_reward = 0
        
        #pros for moving 
        if new_x > self.max_x:
            shaped_reward += new_x - self.max_x
            self.max_x = new_x
            self.no_move_counter = 0

        #Adds kind of like a sudo checkpoint (not truly a checkpoint)
        if new_x > 1800 and not self.checkpoint_hit:  
            shaped_reward += 20
            self.checkpoint_hit = True

        #added bonus for score
    #    if new_score - self.prev_score > 10:
    #            shaped_reward += (new_score - self.prev_score) * 0.005

        #level complete + reward for finishing earlier
        if flag_get:
            shaped_reward += 50
            shaped_reward += max(0, 15 - self.frame_count // 100)  # Max 15 bonus, less if slower


        #if dead or time run out
        if done and not flag_get:
            shaped_reward -= 2

        #if you dont move it closes 
        if new_x <= self.max_x:
            self.no_move_counter += 1
        else:
            self.no_move_counter = 0

        #closes training if it isnt moving 
        if self.no_move_counter > 100:
            shaped_reward -= 1
            done = True
            print("Episode ended early: stuck too long.")

        #If its going too long it ends
        if self.frame_count >= self.max_frames:
            done = True
            print("Episode ended early: frame limit reached.")

        #Force end if life count drops
        if 'life' in info and info['life'] < self.prev_life:
            print("🔄 Life lost! Forcing reset to 1-1.")
            shaped_reward -= 2
            done = True

        self.prev_life = info.get('life', 2)
        self.prev_x = new_x
        self.prev_score = new_score

        return obs, shaped_reward, done, info

# === Action Repeat ===
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

# === Callback ===
class AutoSaveCallback(BaseCallback):
    def __init__(self, save_path, save_freq=1000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            if self.verbose:
                print(f"✅ Autosaved model to {self.save_path} at step {self.n_calls*num_envs}")
        return True

# === Parallel Environment Setup ===
def make_env():
    def _init():
        env = gym_super_mario_bros.make('SuperMarioBros-v3') #the level (i think)

        env = JoypadSpace(env, CUSTOM_MOVEMENT) # my custom movemnet
        env = ActionRepeatWrapper(env, repeat=2) # this is the repeater wrapper
        env = MarioRewardWrapper(env) # applies the wrapper
        env = ResizeObservation(env, (84, 84))  #Resize frame to 84x84
        env = Monitor(env)
        return env
    return _init

if __name__ == '__main__':
    num_envs = 8
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    # === Paths ===
    CHECKPOINT_DIR = "./checkpoints"
    FINAL_MODEL_PATH = "ppo_mario_final"
    LATEST_MODEL_PATH = "ppo_mario_latest"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # === Load Model ===
    model = None
    latest_ckpt = None
    if os.path.exists(f"{LATEST_MODEL_PATH}.zip"):
        model = PPO.load(LATEST_MODEL_PATH, env)
        print("🔁 Resumed from latest autosave")
    elif any(f.endswith('.zip') and 'mario_ppo' in f for f in os.listdir(CHECKPOINT_DIR)):
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.zip') and 'mario_ppo' in f]
        checkpoints.sort(key=lambda x: int(re.search(r'_(\d+)_steps', x).group(1)))
        latest_ckpt = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
        model = PPO.load(latest_ckpt, env)
        print(f"🔁 Resumed from checkpoint: {latest_ckpt}")
    else:
        model = PPO("CnnPolicy", env, verbose=1, n_steps=1024, tensorboard_log="./ppo_mario_logs")
        print("🆕 Starting training from scratch")

    # === Callbacks ===
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=CHECKPOINT_DIR,
        name_prefix="mario_ppo"
    )
    autosave_callback = AutoSaveCallback(LATEST_MODEL_PATH, save_freq=2048, verbose=1)

    # === Train ===
    model.learn(total_timesteps=1_000_000_000, callback=[checkpoint_callback, autosave_callback])

    # === Save final model ===
    model.save(FINAL_MODEL_PATH)
    print("✅ Final model saved.")
