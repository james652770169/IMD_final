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
        """回傳 base, joint1_end, joint2_end, end_effector (全部是畫面座標系)"""
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
        """state_dim=15"""
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
