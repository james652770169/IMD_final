import streamlit as st
st.set_page_config(page_title="國立虎尾科技大學機械設計工程系", layout="wide")

st.title("第二章 Turtlebot3 burger之深度學習分類問題")

st.markdown(""""
以 TurtleBot3（burger/waffle 平台）為移動載具，整合 ROS 與深度學習影像分類，完成「手勢/動作辨識 → 對應移動指令」之即時控制系統。影像分類模型透過 Google Teachable Machine 建立 5 類別（left、right、forward、back/down、stop），並匯出 Keras 模型檔（keras_model.h5）
        """)

# 在這裡添加實驗一的具體內容，如圖表、數據等
st.header("2.1 由Teachable Machine訓練的模型")
st.markdown(""""
模型訓練:類別設計

left：左轉

right：右轉

forward：前進

back/down：後退（或 down 手勢代表後退）

stop：停止（安全動作））
        """)

st.markdown(""""
| Step | 階段                 | Teachable Machine 操作（你在網站上要點哪裡）            | 參數/設定重點（建議值與說明）                                                                                                                                                                                     | 成功判斷（看到什麼才算完成）                                                         | 常見問題 & 解法                                                                                                                                                                                                      |
| ---: | ------------------ | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  1 | 訓練參數設定             | 右側面板點 **Advanced** 展開（Training 設定區）                                                                            | **Epochs**：建議 30–80（常用 50）→ 越高越容易學到特徵，但過高可能過擬合。<br>**Batch Size**：建議 16 或 32 → 16 較穩、32 較快。<br>**Learning Rate**：建議 0.001（常用）→ 太大會震盪、太小會學很慢。<br>**資料量建議**：每類 ≥100 張、類別數量盡量平衡（避免某類太少造成偏誤）。           | 右側參數已設定完成，Training 區塊可開始 Train                                         | **訓練前準備不足**：left/right 容易混淆 → 增加兩類樣本、改變手勢角度/距離、提高背景多樣性。<br>**類別不平衡**：某類樣本很少 → 補到相近數量（stop 也要足夠）。                                                                                                               |
|  2 | 開始訓練               | 點 **Train Model**                                                                                              | 訓練時不用做其他操作；等進度跑完                                                                                                                                                                                    | 顯示 **Model Trained**（或完成提示），可進入即時測試                                    | **Accuracy 很低/一直猜同一類**：通常資料太像或類別不清楚 → 回到資料蒐集補差異（手勢位置、角度、背景、光源）。<br>**訓練完成但效果差**：降低 LR 或增加 Epoch；也可重新整理資料（刪掉模糊/錯標）。                                                                                             |
|  3 | 即時測試（快速驗證）         | 使用 TM 內建 **Preview / Webcam test**（直接對鏡頭做手勢）                                                                   | 測試要涵蓋：不同光線、不同距離、不同手角度；並觀察「信心值」是否穩定                                                                                                                                                                  | 不同手勢時能正確切換類別；stop 能穩定被辨識（建議 stop 信心要高）                                 | **抖動頻繁跳類別**：資料太接近或背景干擾 → 補資料、提高 stop 樣本、讓手勢更標準化。<br>**某兩類一直互相混**：left/right 常見 → 讓動作更具差異（手掌方向/位置），或各自補更多樣本。                                                                                                    |
|  4 | 匯出模型（Keras）        | 點 **Export Model** → 選 **Keras** → 下載                                                                          | 下載後會得到兩個關鍵檔：<br>1) `keras_model.h5`（模型）<br>2) `labels.txt`（類別文字表）<br>※ 檔名大小寫務必確認（有時可能是 `keras_Model.h5`，以你下載到的為準）                                                                                   | 電腦本機資料夾中可看到 `.h5` 與 `labels.txt`                                       | **程式載入失敗（找不到檔案）**：99% 是「路徑或檔名」問題 → 先確認檔案真的存在、檔名大小寫一致，再用程式組相對路徑。                                                                                                                                                |
|  5 | 部署到 ROS 專案（放到正確位置） | 將匯出的檔案放入 ROS package 固定資料夾 | **固定路徑策略（強烈建議）**：模型不要放在亂七八糟的位置，統一放 `model/`。<br>**程式讀取策略**：用「程式檔所在位置」去組路徑（避免你在不同目錄執行導致找不到）。<br>**建置/安裝提醒**：若你用 catkin 安裝到 `install_isolated`，記得也要確保 model 檔會被帶進安裝後的位置（或乾脆在程式中讀取 source 資料夾的 model）。 | 1) `ls gesture_control/model` 能看到兩個檔案。<br>2) 程式能成功 `load_model()` 不報錯。 | **你遇到的典型錯誤**：程式去找 `/install_isolated/lib/model/...` → 代表你用「安裝後路徑」推算錯了。解法：<br>✅ 用 `rospack find gesture_control` 找 package 根目錄，再組 `.../model/keras_model.h5`。<br>✅ 或在 CMake / install 規則把 `model/` 一起安裝到 share。 |

        """)
st.write("輸入圖片由Teachable Machine訓練!")
st.image("Picture/第2章01.png")
st.write("測試訓練結果!")
st.image("Picture/第2章02.png")

st.markdown('### 匯出Keras模型')
code = """
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

            """

with st.expander("點擊展開完整程式碼"):
     st.code(code, language="python")


st.header("2.2 重複第一章Turtlebot3(Burger)之避障與導航實作的步驟")
st.header("2.3 確認匯出模型的資料夾")
st.header("2.4 輸入py檔")
code = """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


YOLOv8 + ROS1 圖片控制 TurtleBot3（Burger）

核心流程：
(1) 訂閱車上相機影像 topic（預設 /usb_cam/image_raw）
(2) 讀取訓練好的 YOLO 權重 best.pt，對每帧影像做偵測推論
(3) 取最可信的一個偵測結果（conf 最大的 box）
(4) 將偵測到的 label（例如 left/right/stop/...）映射到動作（LEFT/RIGHT/FWD/BACK/STOP）
(5) 發布 Twist 到 /cmd_vel，使 TurtleBot3 做出反應
(6) 透過 conf 門檻 + 防抖 + 冷卻時間，避免誤判導致抖動或指令狂發


import os
import time
import math  # 本版雖未用到 rpm 換算，但保留也可（如需擴充 rpm 模式）
import rospy
import cv2
import rospkg

from ultralytics import YOLO
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError


def _split_labels(s: str):
    
    將像 "left,turn left" 這種字串參數，切成 list 並統一小寫：
    例如 -> ["left", "turn left"]
    用來讓 label mapping 可以在 launch/rosrun 端彈性改類別名稱，不必改程式。
    
    return [x.strip().lower() for x in s.split(",") if x.strip()]


class YOLOGestureControl:
    
    預設（最常見）使用 TB3 上的 usb_cam raw 影像：
      Sub: /usb_cam/image_raw  (sensor_msgs/Image)

    若你真的有 compressed 影像，也能切換：
      Sub: /usb_cam/image_raw/compressed (sensor_msgs/CompressedImage) when ~use_compressed:=true

    控制輸出：
    Pub: /cmd_vel (geometry_msgs/Twist)
    

    def __init__(self):
        # -------------------------
        # 1) ROS node 初始化
        # -------------------------
        rospy.init_node("yolo_gesture_control", anonymous=False)

        # -------------------------
        # 2) 讀取參數：package 名稱、模型路徑
        # -------------------------
        # pkg_name 用來讓程式能用 rospkg 找到 package 位置
        self.pkg_name = rospy.get_param("~pkg_name", "gesture_control")

        # model_path 可給：
        # - 絕對路徑：/home/user/.../best.pt
        # - 相對路徑：best.pt（會自動去 <pkg>/model/best.pt 找）
        model_path_param = rospy.get_param("~model_path", "best.pt")
        self.model_path = self.resolve_model_path(model_path_param)

        # 檢查模型檔是否存在
        if not os.path.isfile(self.model_path):
            rospy.logerr("❌ Model not found: %s", self.model_path)
            rospy.logerr("請把 best.pt 放到 <pkg>/model/ 或用 _model_path:=/abs/path/best.pt")
            raise RuntimeError("model file not found")

        # 載入 YOLO 模型（Ultralytics YOLOv8）
        rospy.loginfo("✅ Using model: %s", self.model_path)
        self.model = YOLO(self.model_path)

        # -------------------------
        # 3) topic 設定：影像訂閱與速度指令輸出
        # -------------------------
        # use_compressed=False -> 使用 /usb_cam/image_raw（sensor_msgs/Image）
        # use_compressed=True  -> 使用 /usb_cam/image_raw/compressed（sensor_msgs/CompressedImage）
        self.use_compressed = bool(rospy.get_param("~use_compressed", False))  # ✅ 預設 False（吃 raw）

        self.image_topic = rospy.get_param(
            "~image_topic",
            "/usb_cam/image_raw" if not self.use_compressed else "/usb_cam/image_raw/compressed"
        )
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")

        # -------------------------
        # 4) YOLO 推論參數
        # -------------------------
        # conf_th：信心門檻，太低會誤判多；太高會偵測不到
        self.conf_th = float(rospy.get_param("~conf_th", 0.6))

        # imgsz：YOLO 推論輸入尺寸（常見 640）
        self.imgsz = int(rospy.get_param("~imgsz", 640))

        # -------------------------
        # 5) 防抖/防連發參數（讓車更穩）
        # -------------------------
        # hold_same_label_n：同一個手勢連續出現 N 帧才觸發（避免一兩帧誤判）
        self.hold_same_label_n = int(rospy.get_param("~hold_same_label_n", 3))

        # cooldown_s：觸發一次後冷卻幾秒內不再觸發新動作（避免狂發命令）
        self.cooldown_s = float(rospy.get_param("~cooldown_s", 0.8))

        # -------------------------
        # 6) 控制參數（vel 模式：直接用 /cmd_vel 的線/角速度）
        # -------------------------
        # lin_fwd：前進速度（m/s）
        # lin_back：後退速度（m/s，通常是負值）
        # ang_left：左轉角速度（rad/s，正值）
        # ang_right：右轉角速度（rad/s，負值）
        self.lin_fwd = float(rospy.get_param("~lin_fwd", 0.12))
        self.lin_back = float(rospy.get_param("~lin_back", -0.10))
        self.ang_left = float(rospy.get_param("~ang_left", 0.9))
        self.ang_right = float(rospy.get_param("~ang_right", -0.9))

        # 動作持續時間（秒）
        # 例如左轉一次維持 1 秒就停
        self.t_turn = float(rospy.get_param("~t_turn", 1.0))
        self.t_fwd = float(rospy.get_param("~t_fwd", 1.5))
        self.t_back = float(rospy.get_param("~t_back", 1.2))

        # 速度安全限制（避免速度設太大）
        self.max_lin = float(rospy.get_param("~max_lin", 0.22))  # burger 常用安全上限
        self.max_ang = float(rospy.get_param("~max_ang", 2.0))

        # 是否顯示視窗（顯示 YOLO 偵測框）
        self.show_window = bool(rospy.get_param("~show_window", True))

        # -------------------------
        # 7) label mapping：模型輸出類別名稱 -> 動作代碼
        # -------------------------
        # 你訓練的 best.pt 可能類別叫：left / right / stop / go straight / ...
        # 這裡用參數定義可接受的同義詞（並轉小寫比對）
        self.labels_left = _split_labels(rospy.get_param("~labels_left", "left"))
        self.labels_right = _split_labels(rospy.get_param("~labels_right", "right"))
        self.labels_stop = _split_labels(rospy.get_param("~labels_stop", "stop"))
        self.labels_fwd = _split_labels(rospy.get_param("~labels_forward", "go straight,forward"))
        self.labels_back = _split_labels(rospy.get_param("~labels_back", "back,backward,reverse,start"))

        # -------------------------
        # 8) ROS Publisher/Subscriber & CvBridge
        # -------------------------
        self.pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)
        self.bridge = CvBridge()
        self.frame = None  # 用來保存最新一帧影像（OpenCV bgr8）

        # 根據影像型態選擇不同 callback（raw vs compressed）
        if self.use_compressed:
            rospy.Subscriber(self.image_topic, CompressedImage, self.cb_compressed, queue_size=1)
        else:
            rospy.Subscriber(self.image_topic, Image, self.cb_raw, queue_size=1)

        # -------------------------
        # 9) 非阻塞動作狀態機（避免 rospy.sleep 卡住推論）
        # -------------------------
        # active_action：目前正在執行的動作（LEFT/RIGHT/FWD/BACK 或 None）
        # action_end_t：動作結束時間（time.time() 的秒數）
        # last_trigger_t：上一次觸發動作的時間（用於 cooldown）
        self.active_action = None
        self.action_end_t = 0.0
        self.last_trigger_t = 0.0

        # 防抖狀態：記錄最近一次 action 與連續次數
        self.last_seen_action = None
        self.same_count = 0

        rospy.loginfo("✅ Sub: %s | Pub: %s", self.image_topic, self.cmd_topic)

    # -------------------------
    # 工具：解析模型檔路徑
    # -------------------------
    def resolve_model_path(self, p: str) -> str:
        
        若 p 是絕對路徑 -> 直接回傳
        若 p 是相對路徑（例如 best.pt） -> 組成 <pkg>/model/best.pt
        
        if os.path.isabs(p):
            return p
        rp = rospkg.RosPack()
        pkg_path = rp.get_path(self.pkg_name)
        return os.path.join(pkg_path, "model", p)

    # -------------------------
    # 影像 callback：compressed
    # -------------------------
    def cb_compressed(self, msg: CompressedImage):
        
        將 ROS CompressedImage 轉成 OpenCV BGR 圖
        
        try:
            self.frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge compressed error: %s", e)

    # -------------------------
    # 影像 callback：raw
    # -------------------------
    def cb_raw(self, msg: Image):
        
        將 ROS Image 轉成 OpenCV BGR 圖
        
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge raw error: %s", e)

    # -------------------------
    # 工具：限制值避免超速
    # -------------------------
    def clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    # -------------------------
    # 發布 /cmd_vel（Twist）
    # -------------------------
    def publish_twist(self, lin_x: float, ang_z: float):
        
        lin_x：線速度（m/s）
        ang_z：角速度（rad/s）
        會先做 clamp，避免超過安全上限
        
        t = Twist()
        t.linear.x = self.clamp(lin_x, -self.max_lin, self.max_lin)
        t.angular.z = self.clamp(ang_z, -self.max_ang, self.max_ang)
        self.pub.publish(t)

    def stop_robot(self):
        立即停止：線速度與角速度都設為 0
        self.publish_twist(0.0, 0.0)

    # -------------------------
    # 非阻塞動作：開始一個動作
    # -------------------------
    def start_action(self, action_code: str, duration_s: float):
        
        設定 active_action 與結束時間，之後由 step_action() 在每次迴圈持續發布速度直到到期
        這樣不會阻塞（不會 sleep 卡住推論）
        
        self.active_action = action_code
        self.action_end_t = time.time() + duration_s
        self.last_trigger_t = time.time()
        rospy.loginfo("🚀 Action: %s (%.2fs)", action_code, duration_s)

    # -------------------------
    # 非阻塞動作：每圈維持動作直到到期
    # -------------------------
    def step_action(self):
        
        若目前有 active_action，就持續發布相應速度；
        到期則停止並清空 active_action
        
        if not self.active_action:
            return

        now = time.time()
        if now >= self.action_end_t:
            self.active_action = None
            self.stop_robot()
            return

        # 根據動作代碼發布速度
        if self.active_action == "LEFT":
            self.publish_twist(0.0, self.ang_left)
        elif self.active_action == "RIGHT":
            self.publish_twist(0.0, self.ang_right)
        elif self.active_action == "FWD":
            self.publish_twist(self.lin_fwd, 0.0)
        elif self.active_action == "BACK":
            self.publish_twist(self.lin_back, 0.0)

    # -------------------------
    # 從 YOLO results 中挑 conf 最大的一個偵測
    # -------------------------
    def pick_best(self, results):
        
        YOLO 可能偵測到多個框：
        這裡選 conf 最大的那個（最可信）
        回傳：(label, conf)
        
        if not results or len(results) == 0:
            return None, 0.0

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return None, 0.0

        best_conf = -1.0
        best_cls = None

        for b in r0.boxes:
            conf = float(b.conf[0]) if b.conf is not None else 0.0
            cls_id = int(b.cls[0]) if b.cls is not None else None
            if cls_id is not None and conf > best_conf:
                best_conf = conf
                best_cls = cls_id

        if best_cls is None:
            return None, 0.0

        # model.names: cls_id -> label 字串
        label = self.model.names.get(best_cls, str(best_cls))
        return label, best_conf

    # -------------------------
    # 將模型 label 轉成動作代碼
    # -------------------------
    def label_to_action(self, label: str):
        
        把模型輸出的 label（字串）映射成：
        STOP / LEFT / RIGHT / FWD / BACK / NONE
        （label 先轉小寫再比對）
        
        if label is None:
            return "NONE"

        s = label.strip().lower()

        if s in self.labels_stop:
            return "STOP"
        if s in self.labels_left:
            return "LEFT"
        if s in self.labels_right:
            return "RIGHT"
        if s in self.labels_fwd:
            return "FWD"
        if s in self.labels_back:
            return "BACK"
        return "NONE"

    # -------------------------
    # 主迴圈：推論 + 控制
    # -------------------------
    def run(self):
        rate = rospy.Rate(20)  # 20Hz 控制迴圈
        rospy.loginfo("Running... (Ctrl+C to exit)")

        while not rospy.is_shutdown():
            # 1) 先維持正在執行的動作（非阻塞）
            self.step_action()

            # 2) 還沒收到影像就等待
            if self.frame is None:
                rate.sleep()
                continue

            # 3) YOLO 推論
            results = self.model.predict(self.frame, imgsz=self.imgsz, verbose=False)

            # 4) 取最可信偵測 + 映射動作
            label, conf = self.pick_best(results)
            action = self.label_to_action(label)

            # 5) 可視化顯示（調試用）
            if self.show_window:
                annotated = results[0].plot()
                cv2.putText(
                    annotated,
                    f"{label} ({conf:.2f}) -> {action}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                cv2.imshow("YOLO Gesture Control", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 6) 濾掉無效結果：NONE 或 conf 不足
            if action == "NONE" or conf < self.conf_th:
                self.last_seen_action = None
                self.same_count = 0
                rate.sleep()
                continue

            # 7) 防抖：同一 action 連續 N 帧才觸發
            if action == self.last_seen_action:
                self.same_count += 1
            else:
                self.last_seen_action = action
                self.same_count = 1

            if self.same_count < self.hold_same_label_n:
                rate.sleep()
                continue

            # 8) STOP 永遠最高優先：立刻中斷任何動作並停止
            if action == "STOP":
                self.active_action = None
                self.stop_robot()
                self.last_trigger_t = time.time()
                rate.sleep()
                continue

            # 9) 冷卻時間：避免一直觸發新動作
            if (time.time() - self.last_trigger_t) < self.cooldown_s:
                rate.sleep()
                continue

            # 10) 若目前正在動作中（active_action != None），就不開新動作
            if self.active_action is not None:
                rate.sleep()
                continue

            # 11) 觸發動作（非阻塞）
            if action == "LEFT":
                self.start_action("LEFT", self.t_turn)
            elif action == "RIGHT":
                self.start_action("RIGHT", self.t_turn)
            elif action == "FWD":
                self.start_action("FWD", self.t_fwd)
            elif action == "BACK":
                self.start_action("BACK", self.t_back)

            rate.sleep()

        # 程式退出前清理視窗、停止車
        cv2.destroyAllWindows()
        self.stop_robot()


if __name__ == "__main__":
    try:
        YOLOGestureControl().run()
    except rospy.ROSInterruptException:
        pass


            """
with st.expander("點擊展開完整程式碼"):
         st.code(code, language="python")




st.header("2.5 成果展示")
st.video("videos/video04.mp4")