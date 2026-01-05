import numpy as np
import pyglet

class ArmEnv(object):
    viewer = None
    dt = .1                 # 刷新率
    action_bound = [-1, 1]  # 動作轉動角度限制
    goal = {'x': 100., 'y': 100., 'l': 40} # 藍色目標的初始位置與大小
    state_dim = 9           # 觀測值維度 (距離x,y + 接觸判定 + 3個手臂的sin/cos = 9)
    action_dim = 3          # 動作維度 (3 個馬達)

    def __init__(self):
        # --- 1. 定義三段手臂的長度 ---
        self.arm_info = np.zeros((3, 2))
        self.arm_info[0, 0] = 100   # 第一段手臂長度
        self.arm_info[1, 0] = 100   # 第二段手臂長度
        self.arm_info[2, 0] = 50    # 第三段手臂長度 (新增的)
        
        # 初始化角度 (隨機)
        self.arm_info[:, 1] = self.goal['l'] * np.pi * 2

    def step(self, action):
        done = False
        # 限制動作範圍
        action = np.clip(action, -0.5, 0.5)

        
        # 更新手臂角度
        self.arm_info[:, 1] += action * self.dt
        self.arm_info[:, 1] %= np.pi * 2    # 正規化角度在 0~2pi

        # --- 2. 運動學計算 (Kinematics) ---
        # 取得角度 (tr) 與 長度 (l)
        tr = self.arm_info[:, 1]
        l = self.arm_info[:, 0]

        # 計算第一段末端座標
        a1xy = np.array([np.cos(tr[0]), np.sin(tr[0])]) * l[0]
        
        # 計算第二段末端座標 (基於第一段的位置)
        a2xy = np.array([np.cos(tr[0]+tr[1]), np.sin(tr[0]+tr[1])]) * l[1] + a1xy
        
        # 計算第三段末端座標 (基於第二段的位置)
        a3xy = np.array([np.cos(tr[0]+tr[1]+tr[2]), np.sin(tr[0]+tr[1]+tr[2])]) * l[2] + a2xy

        # 計算「末端點」與「目標」的距離
        dist = np.sqrt(np.sum(np.square(a3xy - np.array([self.goal['x'], self.goal['y']]))))

        # 獎勵函數 (距離越近，分數越高)
        r = -dist

        # 判斷是否接觸到目標 (Touch)
        if dist < self.goal['l']:
            r += 100.
            self.on_goal += 1
            if self.on_goal > 50:
                done = True
        else:
            self.on_goal = 0

        # --- 3. 回傳狀態 (State) ---
        # 狀態包含: 目標距離向量(2), 是否接觸(1), 三個關節的角度sin/cos(6) -> 共 9 個值
        dist_vec = np.array([self.goal['x'], self.goal['y']]) - a3xy
        s = np.concatenate((dist_vec/200, [1. if self.on_goal else 0.],
                            np.cos(tr), np.sin(tr)))
        return s, r, done

    def reset(self):
        self.on_goal = 0
        self.arm_info[:, 1] = self.goal['l'] * np.pi * 2
        
        # Reset 時也要計算一次座標以回傳正確的 State
        tr = self.arm_info[:, 1]
        l = self.arm_info[:, 0]
        a1xy = np.array([np.cos(tr[0]), np.sin(tr[0])]) * l[0]
        a2xy = np.array([np.cos(tr[0]+tr[1]), np.sin(tr[0]+tr[1])]) * l[1] + a1xy
        a3xy = np.array([np.cos(tr[0]+tr[1]+tr[2]), np.sin(tr[0]+tr[1]+tr[2])]) * l[2] + a2xy
        
        dist_vec = np.array([self.goal['x'], self.goal['y']]) - a3xy
        s = np.concatenate((dist_vec/200, [1. if self.on_goal else 0.],
                            np.cos(tr), np.sin(tr)))
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(3)-0.5    # 隨機產生 3 個馬達的動作

# --- 視覺化視窗 (Viewer) ---
class Viewer(pyglet.window.Window):
    bar_thc = 5 # 手臂寬度

    def __init__(self, arm_info, goal):
        # 建立視窗, 啟用滑鼠事件
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm')
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.goal_info = goal
        self.center_coord = np.array([200, 200]) # 手臂基座位置

        self.batch = pyglet.graphics.Batch()
        
        # 藍色目標方塊 (Target)
        self.point = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))

        # 紅色手臂 1
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250, 250, 300, 260, 300, 260, 250]), # 初始佔位符
            ('c3B', (249, 86, 86) * 4,))
        
        # 紅色手臂 2
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150, 100, 160, 200, 160, 200, 150]), 
            ('c3B', (249, 86, 86) * 4,))

        # 紅色手臂 3 (新增)
        self.arm3 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150, 100, 160, 200, 160, 200, 150]), 
            ('c3B', (249, 86, 86) * 4,))

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        # 1. 更新目標方塊位置 (跟隨滑鼠)
        # 這段會讓藍色方塊移動到 self.goal_info 更新後的座標
        self.point.vertices = np.array([
            self.goal_info['x'] - self.goal_info['l'] / 2, self.goal_info['y'] - self.goal_info['l'] / 2,
            self.goal_info['x'] - self.goal_info['l'] / 2, self.goal_info['y'] + self.goal_info['l'] / 2,
            self.goal_info['x'] + self.goal_info['l'] / 2, self.goal_info['y'] + self.goal_info['l'] / 2,
            self.goal_info['x'] + self.goal_info['l'] / 2, self.goal_info['y'] - self.goal_info['l'] / 2
        ]).astype(int).tolist() # 轉成 list 避免 Python 3 錯誤

        # 2. 計算手臂座標
        (a1l, a2l, a3l) = self.arm_info[:, 0]
        (a1r, a2r, a3r) = self.arm_info[:, 1]
        
        # Arm 1 End
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l
        a1xy = self.center_coord + a1xy_
        
        # Arm 2 End
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l
        a2xy = a1xy + a2xy_

        # Arm 3 End
        a3xy_ = np.array([np.cos(a1r+a2r+a3r), np.sin(a1r+a2r+a3r)]) * a3l
        a3xy = a2xy + a3xy_

        # 3. 更新手臂繪圖 (解決斷裂問題的關鍵)
        # 在 Python 3 中，必須將 numpy array 轉成 list，Pyglet 才能正確更新頂點
        
        # Update Arm 1
        self.arm1.vertices = np.concatenate((
            self.center_coord - np.array([5, 0]), 
            self.center_coord + np.array([5, 0]),
            a1xy + np.array([5, 0]), 
            a1xy - np.array([5, 0])
        )).astype(int).tolist()

        # Update Arm 2
        self.arm2.vertices = np.concatenate((
            a1xy - np.array([5, 0]), 
            a1xy + np.array([5, 0]),
            a2xy + np.array([5, 0]), 
            a2xy - np.array([5, 0])
        )).astype(int).tolist()

        # Update Arm 3
        self.arm3.vertices = np.concatenate((
            a2xy - np.array([5, 0]), 
            a2xy + np.array([5, 0]),
            a3xy + np.array([5, 0]), 
            a3xy - np.array([5, 0])
        )).astype(int).tolist()

    # --- 關鍵：恢復滑鼠互動 ---
    def on_mouse_motion(self, x, y, dx, dy):
        # 當滑鼠移動時，更新目標座標
        self.goal_info['x'] = x
        self.goal_info['y'] = y

if __name__ == '__main__':
    # 測試區塊
    env = ArmEnv()
    while True:
        env.render()
        env.step(env.sample_action())