# main.py
from env import ArmEnv
from rl import DDPG
import numpy as np


# 訓練參數
MAX_EPISODES = 3000
MAX_EP_STEPS = 700
#ON_TRAIN = True
ON_TRAIN = False


# 建立環境與 RL
def train():
    # 訓練：目標每回合隨機（學追任意位置）
    env = ArmEnv(allow_mouse_goal=False, random_goal_on_reset=True)

    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound

    rl = DDPG(a_dim, s_dim, a_bound)

    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.0

        for j in range(MAX_EP_STEPS):
            # 想看訓練畫面可以打開（會變慢）
            # env.render()

            a = rl.choose_action(s)

            # 加一點探索噪聲（更容易學到）
            a = np.clip(np.random.normal(a, 0.2), -1, 1)

            s_, r, done = env.step(a)
            rl.store_transition(s, a, r, s_)

            ep_r += r

            if rl.memory_full:
                rl.learn()

            s = s_

            if done or j == MAX_EP_STEPS - 1:
                print('Ep: %i | %s | ep_r: %.2f | step: %i' %
                      (i, 'done' if done else '---', ep_r, j))
                break

    rl.save()
    print("[DONE] training finished, params saved.")

def eval():
    # 展示：目標跟滑鼠（policy 追滑鼠）
    env = ArmEnv(allow_mouse_goal=True, random_goal_on_reset=False)

    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound

    rl = DDPG(a_dim, s_dim, a_bound)
    rl.restore()

    s = env.reset()
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done = env.step(a)

if __name__ == '__main__':
    if ON_TRAIN:
        train()
    else:
        eval()
