import streamlit as st
st.set_page_config(page_title="國立虎尾科技大學機械設計工程系", layout="wide")


st.title("第四章 RL建模與訓練")
st.write("完成three-link planar manipulator之RL強化學習訓練模擬")

st.markdown(""""
以「三連桿平面機械手臂（Three-link Planar Manipulator）」為研究對象，結合強化學習（Reinforcement Learning, RL）方法，建構一套可於二維平面中自主學習並追蹤目標之控制系統。系統以 Python 為主要開發語言，並透過模組化設計區分為三大核心元件：

環境模型（Environment, env.py）

強化學習演算法（RL Agent, rl.py）

訓練與執行流程控制（Main Controller, main.py）

此架構能清楚分離「物理模擬」、「學習策略」與「實驗流程」，提升系統的可維護性與可擴充性。

        """)


st.markdown(""""
| 檔案名稱        | 檔案角色定位             | 主要功能說明                                    | 關鍵內容 / 類別                                                                                                   | 使用與操作方式                                     |
| ----------- | ------------------ | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **env.py**  | 環境模型（Environment）  | 建立三連桿平面機械手臂的模擬環境，定義狀態、動作、獎勵與終止條件，並提供即時視覺化 | `ArmEnv` 類別<br>• 三連桿長度設定<br>• Forward Kinematics<br>• reward 設計<br>• state 設計（15 維）<br>• pyglet 視覺化         | 由 `main.py` 呼叫<br>不可單獨訓練<br>可用於環境測試（render） |
| **rl.py**   | 強化學習演算法（Agent）     | 實作 DDPG 強化學習模型，負責動作決策與策略更新                | `DDPG` 類別<br>• Actor / Critic Network<br>• Replay Buffer<br>• Soft Target Update<br>• TensorFlow v1 Session | 由 `main.py` 建立物件<br>負責學習與推論<br>不可獨立運行       |
| **main.py** | 主程式控制器（Controller） | 控制整體流程，整合環境與 RL，負責訓練與展示模式切換               | `train()`<br>`eval()`<br>ON_TRAIN 旗標                                                                        | 直接執行的入口程式<br>`python main.py`               |

        """)


st.markdown("""
            <p style="text-indent: 0em; line-height: 1.8; font-size: 30px; text-align: justify; font-weight: bold;">
                3.1程式碼 main.py  訓練與展示流程控制
            
建立 env                          

建立 DDPG

決定是「訓練」還是「展示」

控制 episode / step
                </p>
                """, unsafe_allow_html=True)    
code = """
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

        """
with st.expander("點擊展開完整程式碼"):
     st.code(code, language="python")


     st.markdown(""""
| 模式   | 設定                 | 功能說明                      |
| ---- | ------------------ | ------------------------- |
| 訓練模式 | `ON_TRAIN = True`  | 目標固定，加入探索噪聲，進行 RL 訓練並儲存模型 |
| 展示模式 | `ON_TRAIN = False` | 載入已訓練模型，機械手臂即時追蹤滑鼠目標      |


        """)



st.markdown("""
            <p style="text-indent: 0em; line-height: 1.8; font-size: 30px; text-align: justify; font-weight: bold;">
                3.2程式碼 env.py  建立 3 連桿平面機械手臂 

            定義 狀態（state）            

            定義 動作（action）           

            定義 獎勵（reward）           

            視覺化顯示（pyglet）         
            決定「是否完成任務（done）」    
                </p>
                """, unsafe_allow_html=True)    
code = """
# env.py
import numpy as np
import pyglet


class ArmEnv(object):
    viewer = None
    dt = 0.1   # 時間步長，影響角度更新速度
    action_bound = [-1, 1]   # 動作範圍（角速度限制）
    action_dim = 3   # 三個關節
    state_dim = 15  # dist_vec(2) + dist(1) + ee_vel(2) + on_goal(1) + cos(3)+sin(3) + prev_action(3)

    def __init__(self, allow_mouse_goal=False, random_goal_on_reset=True):
        # === 視窗尺寸與 base 固定 ===
        self.W, self.H = 400, 400
        self.base = np.array([200., 200.], dtype=np.float32)

        # === 目標方塊大小 50 ===
        self.goal = {'x': 100., 'y': 100., 'l': 50.0}

        # 控制模式
        self.allow_mouse_goal = allow_mouse_goal          # True: 目標跟著滑鼠
        self.random_goal_on_reset = random_goal_on_reset  # True: reset 時目標隨機（訓練用）

        # 三連桿：長度 + 角度
        self.arm_info = np.zeros((3, 2), dtype=np.float32)
        self.arm_info[0, 0] = 100.0
        self.arm_info[1, 0] = 100.0
        self.arm_info[2, 0] = 50.0

        # 初始化角度（建議隨機，讓探索更好）
        self.arm_info[:, 1] = np.random.uniform(0, 2*np.pi, size=3).astype(np.float32)

        # for termination + state
        self.on_goal = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_ee_pos = self._get_ee_pos()
        self.prev_dist = self._dist_to_goal(self.prev_ee_pos)

    # ===================== Kinematics helpers =====================
    def _get_joint_positions(self):
        回傳 base, joint1_end, joint2_end, end_effector (全部是畫面座標系)
        tr = self.arm_info[:, 1]
        l = self.arm_info[:, 0]

        # 注意：這裡是相對 base 的向量，再加上 base 得到畫面座標
        p0 = self.base.copy()
        p1 = p0 + np.array([np.cos(tr[0]), np.sin(tr[0])], dtype=np.float32) * l[0]
        p2 = p1 + np.array([np.cos(tr[0] + tr[1]), np.sin(tr[0] + tr[1])], dtype=np.float32) * l[1]
        p3 = p2 + np.array([np.cos(tr[0] + tr[1] + tr[2]), np.sin(tr[0] + tr[1] + tr[2])], dtype=np.float32) * l[2]
        return p0, p1, p2, p3

    def _get_ee_pos(self):
        return self._get_joint_positions()[-1]

    def _dist_to_goal(self, ee_pos):
        g = np.array([self.goal['x'], self.goal['y']], dtype=np.float32)
        return float(np.linalg.norm(ee_pos - g))

    def _get_state(self):
        state_dim=15
        tr = self.arm_info[:, 1].astype(np.float32)
        ee = self._get_ee_pos()
        g = np.array([self.goal['x'], self.goal['y']], dtype=np.float32)

        dist_vec = (g - ee) / 200.0                 # (2,)
        dist = np.linalg.norm(g - ee) / 200.0       # (1,)
        ee_vel = (ee - self.prev_ee_pos) / 20.0     # (2,)  (簡單正規化)
        touch = 1.0 if self.on_goal > 0 else 0.0    # (1,)
        c = np.cos(tr)                               # (3,)
        s = np.sin(tr)                               # (3,)
        pa = self.prev_action                        # (3,)

        state = np.concatenate([
            dist_vec.astype(np.float32),             # 2
            np.array([dist], dtype=np.float32),      # 1
            ee_vel.astype(np.float32),               # 2
            np.array([touch], dtype=np.float32),     # 1
            c.astype(np.float32),                    # 3
            s.astype(np.float32),                    # 3
            pa.astype(np.float32)                    # 3
        ])
        return state

    # ===================== RL interface =====================
    def step(self, action):
        done = False

        # 動作限制（建議小一點更穩）
        action = np.clip(action, -0.5, 0.5).astype(np.float32)

        # 記錄 prev_action（放進 state）
        self.prev_action = action.copy()

        # 更新角度
        self.arm_info[:, 1] += action * self.dt
        self.arm_info[:, 1] %= (2 * np.pi)

        # 末端位置、距離
        ee = self._get_ee_pos()
        dist = self._dist_to_goal(ee)

        # ===== 更穩的 reward：鼓勵「距離變近」 =====
        # 變近給正分，變遠給負分（比 -dist 穩很多）
        r = (self.prev_dist - dist) * 20.0
        self.prev_dist = dist

        # 末端速度也稍微懲罰，避免亂甩（更穩）
        ee_vel = np.linalg.norm(ee - self.prev_ee_pos)
        r -= 0.01 * ee_vel

        # 判斷命中
        if dist < self.goal['l']:
            r += 5.0
            self.on_goal += 1
            if self.on_goal >= 50:
                done = True
                r += 50.0
        else:
            self.on_goal = 0

        # 更新 prev ee
        self.prev_ee_pos = ee.copy()

        s = self._get_state()
        return s, float(r), done

    def reset(self):
        self.on_goal = 0
        self.prev_action = np.zeros(3, dtype=np.float32)

        # reset 角度（建議隨機）
        self.arm_info[:, 1] = np.random.uniform(0, 2*np.pi, size=3).astype(np.float32)

        # 訓練時：目標隨機（目標大小固定 50）
        if self.random_goal_on_reset and (not self.allow_mouse_goal):
            margin = 70  # 避免太靠邊
            self.goal['x'] = float(np.random.uniform(margin, self.W - margin))
            self.goal['y'] = float(np.random.uniform(margin, self.H - margin))
            self.goal['l'] = 50.0

        self.prev_ee_pos = self._get_ee_pos()
        self.prev_dist = self._dist_to_goal(self.prev_ee_pos)

        return self._get_state()

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal, self.base, self.allow_mouse_goal, self.W, self.H)
        self.viewer.render()

    def sample_action(self):
        return (np.random.rand(3).astype(np.float32) - 0.5)


# ===================== Viewer =====================
class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, goal, base, allow_mouse_goal, W, H):
        super(Viewer, self).__init__(width=W, height=H, resizable=False, caption='Arm')
        pyglet.gl.glClearColor(1, 1, 1, 1)

        self.arm_info = arm_info
        self.goal_info = goal
        self.base = base.astype(np.float32)
        self.allow_mouse_goal = allow_mouse_goal

        self.batch = pyglet.graphics.Batch()

        # 目標方塊
        self.point = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [0, 0, 0, 0, 0, 0, 0, 0]),
            ('c3B', (86, 109, 249) * 4)
        )

        # 三段手臂
        self.arm1 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [0]*8), ('c3B', (249, 86, 86)*4))
        self.arm2 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [0]*8), ('c3B', (249, 86, 86)*4))
        self.arm3 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [0]*8), ('c3B', (249, 86, 86)*4))

    def render(self):
        self._update()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update(self):
        # 更新目標方塊
        x, y, l = self.goal_info['x'], self.goal_info['y'], self.goal_info['l']
        self.point.vertices = np.array([
            x - l/2, y - l/2,
            x - l/2, y + l/2,
            x + l/2, y + l/2,
            x + l/2, y - l/2,
        ], dtype=np.int32).tolist()

        # 計算關節位置（用 env 同一套 forward kinematics）
        a1l, a2l, a3l = self.arm_info[:, 0]
        a1r, a2r, a3r = self.arm_info[:, 1]
        p0 = self.base

        p1 = p0 + np.array([np.cos(a1r), np.sin(a1r)], dtype=np.float32) * a1l
        p2 = p1 + np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)], dtype=np.float32) * a2l
        p3 = p2 + np.array([np.cos(a1r+a2r+a3r), np.sin(a1r+a2r+a3r)], dtype=np.float32) * a3l

        t = self.bar_thc

        # 用「連線向量」決定厚度方向（更不會亂飄/斷裂）
        def quad(pA, pB):
            v = pB - pA
            nv = np.linalg.norm(v) + 1e-6
            v = v / nv
            n = np.array([-v[1], v[0]], dtype=np.float32) * t
            pts = np.concatenate([pA - n, pA + n, pB + n, pB - n])
            return pts.astype(np.int32).tolist()

        self.arm1.vertices = quad(p0, p1)
        self.arm2.vertices = quad(p1, p2)
        self.arm3.vertices = quad(p2, p3)

    def on_mouse_motion(self, x, y, dx, dy):
        if not self.allow_mouse_goal:
            return
        self.goal_info['x'] = float(x)
        self.goal_info['y'] = float(y)


if __name__ == '__main__':
    # demo：滑鼠控制目標
    env = ArmEnv(allow_mouse_goal=True, random_goal_on_reset=False)
    s = env.reset()
    while True:
        env.render()
        s, r, done = env.step(env.sample_action())


        """
with st.expander("點擊展開完整程式碼"):
     st.code(code, language="python")



     st.markdown("""
            | 項目   | 說明                               |
| ---- | -------------------------------- |
| 模擬對象 | 三連桿平面機械手臂（3R Planar Manipulator） |
| 空間限制 | 二維平面（2D）                         |
| 基座位置 | 固定於畫面中心 (200, 200)               |
| 目標方塊 | 固定大小 l = 50，可選擇固定或滑鼠追蹤           |
| 動作空間 | 3 維連續動作（各關節角速度）                  |
| 狀態空間 | 15 維（距離、接觸、角度 sin/cos、末端速度、前一動作） |
| 終止條件 | 末端持續停留於目標區域超過 50 steps           |
| 視覺化  | 使用 pyglet 顯示手臂與目標                |

                """, unsafe_allow_html=True)    




st.markdown("""
            <p style="text-indent: 0em; line-height: 1.8; font-size: 30px; text-align: justify; font-weight: bold;">
                3.3程式碼rl.py  DDPG 強化學習演算法 
            建立 Actor / Critic 神經網路

            維護 replay buffer

            更新策略（policy）

            儲存 / 載入模型
                </p>
                """, unsafe_allow_html=True)    
code = """
# rl.py
import numpy as np
import tensorflow as tf

tf1 = tf.compat.v1
tf1.disable_eager_execution()  # 強制走 TF1 graph/session

LR_A = 0.001
LR_C = 0.001
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 30000
BATCH_SIZE = 32


    Deep Deterministic Policy Gradient
    用於連續動作空間（關節角速度控制）
    
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False

        self.a_dim, self.s_dim = a_dim, s_dim
        self.a_bound = a_bound[1]  # env.action_bound = [-1, 1]

        self.sess = tf1.Session()

        self.S  = tf1.placeholder(tf.float32, [None, s_dim], name='s')
        self.S_ = tf1.placeholder(tf.float32, [None, s_dim], name='s_')
        self.R  = tf1.placeholder(tf.float32, [None, 1],   name='r')

        with tf1.variable_scope('Actor'):
            self.a = self._build_a(self.S,  scope='eval',   trainable=True)
            a_     = self._build_a(self.S_, scope='target', trainable=False)

        with tf1.variable_scope('Critic'):
            q  = self._build_c(self.S,  self.a, scope='eval',   trainable=True)
            q_ = self._build_c(self.S_, a_,     scope='target', trainable=False)
           
         # 收集參數 
        self.ae_params = tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        self.soft_replace = [
            [
                tf1.assign(ta, (1 - TAU) * ta + TAU * ea),
                tf1.assign(tc, (1 - TAU) * tc + TAU * ec),
            ]
            for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)
        ]

        q_target = self.R + GAMMA * q_
        td_error = tf1.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = -tf.reduce_mean(q)
        self.atrain = tf1.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf1.global_variables_initializer())
    由 Actor 網路輸出動作
    def choose_action(self, s):  
        return self.sess.run(self.a, feed_dict={self.S: s[None, :]})[0]
    從 Replay Buffer 取樣並學習
    def learn(self):
        self.sess.run(self.soft_replace)

        idx = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[idx, :]
        bs  = bt[:, :self.s_dim]
        ba  = bt[:, self.s_dim:self.s_dim + self.a_dim]
        br  = bt[:, -self.s_dim - 1:-self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, feed_dict={self.S: bs})
        self.sess.run(self.ctrain, feed_dict={self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
    儲存經驗
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:
            self.memory_full = True

    def _build_a(self, s, scope, trainable):
        with tf1.variable_scope(scope):
            x = tf.keras.layers.Dense(300, activation='relu', trainable=trainable, name='l1')(s)
            a = tf.keras.layers.Dense(self.a_dim, activation='tanh', trainable=trainable, name='a')(x)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf1.variable_scope(scope):
            x = tf.concat([s, a], axis=1)
            x = tf.keras.layers.Dense(300, activation='relu', trainable=trainable, name='l1')(x)
            q = tf.keras.layers.Dense(1, trainable=trainable, name='q')(x)
            return q

    def save(self):
        saver = tf1.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)

    def restore(self):
        saver = tf1.train.Saver()
        saver.restore(self.sess, './params')


        """
with st.expander("點擊展開完整程式碼"):
     st.code(code, language="python")


     st.markdown("""
 | 模組             | 功能說明                                |
| -------------- | ----------------------------------- |
| Actor Network  | 根據狀態輸出連續關節動作                        |
| Critic Network | 評估狀態–動作對的 Q 值                       |
| Replay Buffer  | 儲存互動經驗以打破時間相關性                      |
| Target Network | 穩定訓練（Soft Update）                   |
| Optimizer      | Adam Optimizer                      |
| TensorFlow 模式  | TF 2.x + `compat.v1`（Session-based） |

                """, unsafe_allow_html=True)    


st.header("3.4 成果影片")
st.video("videos/video02.mp4")
st.video("videos/video03.mp4")
