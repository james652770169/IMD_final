import streamlit as st
st.set_page_config(page_title="國立虎尾科技大學機械設計工程系", layout="wide")

st.title("第一章 利用burger或waffler平台，完成自主導航與避障")



# 在這裡添加實驗一的具體內容，如圖表、數據等
st.header("1.1 系統與環境")
st.write("1.平台:TurtleBot3 burger")
st.write("2.架構:PC(ROS Master + RViz + 演算法）＋ Pi(車端 bringup + 感測器）")
st.write("3.主要 ROS 套件:turtlebot3、gmapping / slam、navigation、teleop、rviz、DWA local planner")

st.header("1.2 目標")
st.write("(a) 建立地圖(SLAM)")
st.write("(b) 儲存地圖(map_server)")
st.write("(c) 載入地圖進行定位(AMCL)+路徑規劃(move_base)")
st.write("(d) 在導航過程中即時避障(costmap + DWA planner)")

st.header("1.3 實作流程")
st.write("(A) PC 端建置與套件!")
st.write(" 套件清單!")
st.markdown("""1) TurtleBot3 基本套件

ros-noetic-turtlebot3 



ros-noetic-turtlebot3-msgs 



2) 建圖（SLAM）（gmapping 與 hector 與 slam-gmapping，可依需求選用；流程中常用 gmapping）

ros-noetic-gmapping 



ros-noetic-slam-gmapping 



ros-noetic-hector-slam 



3) 視覺化

ros-noetic-rviz 



4) 導航與避障（規劃器/導航包）

ros-noetic-turtlebot3-navigation 



ros-noetic-dwa-local-planner（助教檔用 apt-get 安裝）



5) 鍵盤遙控（建圖時需要開車掃描）

ros-noetic-turtlebot3-teleop  """, unsafe_allow_html=True)

st.header(" 前置作業（PC：建置環境 / 安裝套件 / 網路確認 / ROS Master）")



st.markdown("""| Step    | 在哪裡做    | Terminal | 指令/操作                                                                | 成功判斷（看到什麼）                             | 備註/常見錯誤                                               |
| ---------- | ------- | -------: | -------------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------- |
| 1-1        | PC      |       T1 | 建立 `catkin_ws`、`src` 資料夾                                             | 目錄存在                                   | 助教檔列出要建立 catkin_ws 與 src。                             |
| 1-2        | PC      |       T1 | `git clone https://github.com/ROBOTIS-GIT/turtlebot3.git`            | `src/turtlebot3/` 出現                   | 依助教檔流程 clone turtlebot3。                              |
| 1-3        | PC      |       T2 | 安裝題目1需要套件（turtlebot3 / gmapping / rviz / dwa / navigation / teleop…） | apt 安裝完成無 error                        | 助教檔「套件安裝」完整清單在同一行。                                    |
| 1-4        | PC      |       T1 | `catkin_make`                                                        | 編譯成功、無紅字 error                         | 助教檔有要求 `catkin_make`。                                 |
| 1-5        | PC      |       T1 | `ping <車端IP>`（助教示例）                                                  | 有回應、封包不掉                               | 助教檔示例 `ping ...` 用於確認連線。                              |
| 1-6        | PC      |       T3 | `roscore`                                                            | roscore 持續跑、不要關                        | 助教檔指定在某終端開 roscore。                                   |
| 1-7(建議)  | PC & Pi |        — | **設定 ROS 網路變數**：PC 設 `ROS_MASTER_URI` 指向 PC；Pi 也指向 PC                | Pi 不再出現 `unable to contact ROS master` | 卡最久的就是 Master URI 指錯（如 192.168.1.142）。此步驟是你實作經驗必加。 |

""", unsafe_allow_html=True) 
            


st.header("進入車端掃地圖（Pi：bringup；PC：SLAM + teleop；存地圖）")
st.markdown("""| Step | 在哪裡做  | Terminal | 指令/操作                                                      | 成功判斷（看到什麼）                    | 備註/常見錯誤                                                               |
| ---- | ----- | -------: | ---------------------------------------------------------- | ----------------------------- | --------------------------------------------------------------------- |
| 2-1  | PC→Pi |       T4 | `ssh pi@192.168.1.199`                                     | 進入 `pi@raspberrypi:~$`        | 助教檔提供 ssh 與密碼提示。                                                      |
| 2-2  | Pi    |       T4 | `roslaunch turtlebot3_bringup turtlebot3_robot.launch`     | 節點持續跑；PC 端可看到 `/scan` 等 topic | 助教檔指定 bringup 指令。（若此步報 Master 連不到：回去檢查 PC roscore 與 ROS_MASTER_URI）   |
| 2-3  | PC    |       T5 | `export TURTLEBOT3_MODEL=burger`                           | 環境變數設定完成                      | 助教檔在啟動 SLAM 前設定 model。                                                |
| 2-4  | PC    |       T5 | `roslaunch turtlebot3_slam turtlebot3_slam.launch`         | RViz 地圖開始生成；`/map` 出現更新       | 助教檔指定 gmapping SLAM。（若 RViz 顯示 map frame 不存在：通常是 bringup 或 /scan 沒進來） |
| 2-5  | PC    |       T6 | `export TURTLEBOT3_MODEL=burger`                           | OK                            | 助教檔在 teleop 前也有再次設定。                                                  |
| 2-6  | PC    |       T6 | `roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch` | 鍵盤可控車移動；地圖越掃越完整               | 助教檔指定 teleop 指令。（若你 Noetic 套件命名不同：以 `rospack find` 查到的為準）             |
| 2-7  | PC    |  T6（或新開） | `rosrun map_server map_saver -f ~/map`                     | 產生 `~/map.pgm`、`~/map.yaml`   | 助教檔指定 map_saver。                                                      |
| 2-8  | PC    |       T7 | `CTRL + C` 停止建圖（SLAM/teleop）                               | SLAM 停止、回到 prompt             | 助教檔標示 CTRL+C。                                                         |

""", unsafe_allow_html=True) 

st.header("利用掃出來的地圖進行避障（PC：載入地圖 + AMCL 定位 + move_base 規劃）!")
st.markdown("""| Step    | 在哪裡做     |       Terminal | 指令/操作                                                                                   | 成功判斷（看到什麼）                  | 備註/常見錯誤                                                                              |
| ------- | -------- | -------------: | --------------------------------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------ |
| 3-1     | Pi       | T4（bringup 維持） | **確認 bringup 還在跑**                                                                      | 車端仍持續跑節點                    | navigation 必須有車端 `/scan`、底盤控制才會動（bringup 不能關）。                                       |
| 3-2     | PC       |       T6（新開也可） | `roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/map.yaml` | RViz 顯示地圖；AMCL/move_base 啟動 | 助教檔指定 navigation（map_file 帶入 yaml）。                                                  |
| 3-3     | PC（RViz） |              — | RViz 點 **2D Pose Estimate** 設定初始位置                                                      | 雷射點與地圖對齊、定位穩定               | 若定位飄：多點幾次初始姿態，或把車移到特徵明顯處（牆角/走廊）                                                      |
| 3-4     | PC（RViz） |              — | RViz 點 **2D Nav Goal** 下目標點                                                             | 車開始走；遇障礙會繞或停下再規劃            | 若出現 `DWA planner failed`/`can't rotate`：通常是車太靠近障礙或 costmap 覺得周圍都不可行 → 把車移到空曠處再下 goal |
| 3-5（驗收） | PC       |             任一 | 監看導航狀態（選用）                                                                              | goal 進行中/成功/失敗狀態變化          | 可用 `rostopic echo /move_base/status` 快速驗證是否真的有在規劃與執行                                 |

""", unsafe_allow_html=True) 


st.header("1.5 成果照片及影片")
st.image("Picture/第1章01.png")
st.image("Picture/第1章02.png")
st.video("videos/video01.mp4")



              