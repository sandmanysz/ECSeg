import numpy as np
import cv2
import time
import random
from stable_baselines3 import PPO

ppo_agent = PPO.load("ppo_checkpoint/ppo_offloading_6400_3.zip", device="cpu")

# 设置固定的更新间隔（单位：ms）
RAW_EDGE_UPDATE_INTERVAL = 60  # 每 60ms 运行一次循环
PERIOD = 17  # 每 17 帧重新决策

# 加载数据
raw_data_images = np.load("raw_data_half.npy")
edge_seg_images = np.load("edge_segmentation_results_half.npy")
cloud_seg_images = np.load("cloud_segmentation_results_half.npy")
image_latency = np.load("latency_sample_v2.npy")

# 计算最短帧数，防止数组越界
min_frames = min(raw_data_images.shape[0], edge_seg_images.shape[0], cloud_seg_images.shape[0])

# 记录当前的 frame index
frame_index = 0
latency_index = 0
current_latency = image_latency[0] if len(image_latency) > 0 else 100  # 初始化默认 latency
avg_euclidean = 0.0
latency_buffer = []  # 存储当前 PERIOD 的所有 latency
average_latency = current_latency  # 记录上一个 PERIOD 的平均 latency

# Cloud 任务队列（FIFO 但只处理最新任务）
cloud_queue = []
latest_cloud_frame = -1  # 记录 Cloud 结果已经返回的最新帧
current_cloud_task = None  # 记录当前正在处理的 Cloud 任务

# **使用 DISOpticalFlow**
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
dis_2 = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

# **创建窗口**
# cv2.namedWindow("Raw Data", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Edge Segmentation", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Cloud Segmentation", cv2.WINDOW_NORMAL)
# cv2.namedWindow("ECSeg", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("ECSeg", 1024, 512)
# cv2.resizeWindow("Raw Data", 1024, 512)
# cv2.resizeWindow("Edge Segmentation", 1024, 512)
# cv2.resizeWindow("Cloud Segmentation", 1024, 512)
cv2.namedWindow("All Display", cv2.WINDOW_NORMAL)
cv2.resizeWindow("All Display", 2048, 1024)  # 可根据需求调整大小
print("开始模拟视频播放（按 ESC 退出）。")

start_time = time.time()
first_frame_gray = cv2.cvtColor(raw_data_images[0], cv2.COLOR_BGR2GRAY)


while True:
    loop_start_time = time.time()

    # 计算当前时间（单位：毫秒）
    elapsed_time = (loop_start_time - start_time) * 1000

    # **1. 计算上一个 PERIOD 的平均 latency**
    if frame_index % PERIOD == 0 and frame_index > 0:

        average_latency = np.mean(latency_buffer) if latency_buffer else current_latency
        latency_buffer = []  # 清空 buffer，开始新周期

    # **2. 第一个 PERIOD 强制使用 Edge，之后使用 PPO**
    if frame_index < PERIOD:
        decision = "Edge"
    elif frame_index % PERIOD == 0:
        print(f"Avg Euclidean: {avg_euclidean}, Avg Latency: {average_latency}")
        observation = np.array([avg_euclidean/300, (average_latency-22)/2961]).reshape(1, -1)  # 观察值 (1,2)# /6/45
        action, _ = ppo_agent.predict(observation, deterministic=True)
        decision = "Edge" if action == 0 else "Cloud"

    # **3. Raw & Edge 每 60ms 更新**
    if elapsed_time >= (frame_index + 1) * RAW_EDGE_UPDATE_INTERVAL:
        frame_index += 1
        # if frame_index >= min_frames:
        #     break
        if frame_index >= min_frames:
            frame_index = 0
            latency_index = 0
            current_latency = image_latency[0] if len(image_latency) > 0 else 100
            avg_euclidean = 0.0
            latency_buffer = []
            average_latency = current_latency
            cloud_queue = []
            latest_cloud_frame = -1
            current_cloud_task = None
            first_frame_gray = cv2.cvtColor(raw_data_images[0], cv2.COLOR_BGR2GRAY)
            start_time = time.time()  # ⭐重新计时，避免时间太大
            continue

        # **4. Cloud 任务进入队列**
        if latency_index < len(image_latency):
            current_latency = image_latency[latency_index]
            latency_index += 1
        else:
            current_latency = image_latency[-1]

        latency_buffer.append(current_latency)  # 存入 buffer 供下一个 PERIOD 计算
        cloud_queue.append((frame_index, elapsed_time + current_latency))

    # **5. Cloud 任务处理逻辑**
    if current_cloud_task is None and cloud_queue:
        current_cloud_task = cloud_queue.pop(0)

    if current_cloud_task:
        processing_frame, expected_return_time = current_cloud_task
        if elapsed_time >= expected_return_time:
            latest_cloud_frame = processing_frame
            current_cloud_task = None
            if cloud_queue:
                current_cloud_task = cloud_queue[-1]
                cloud_queue.clear()

    # **6. Cloud 结果时间补偿 (DIS Optical Flow)**
    if latest_cloud_frame != -1:
        raw_prev = cv2.cvtColor(raw_data_images[latest_cloud_frame], cv2.COLOR_BGR2GRAY)
        raw_curr = cv2.cvtColor(raw_data_images[frame_index], cv2.COLOR_BGR2GRAY)
        cloud_prev = cloud_seg_images[latest_cloud_frame]

        h, w = cloud_prev.shape[:2]
        small_size = (w // 2, h // 2)
        raw_prev_small = cv2.resize(raw_prev, small_size)
        raw_curr_small = cv2.resize(raw_curr, small_size)
        flow_small = dis.calc(raw_prev_small, raw_curr_small, None)
        flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
        flow *= 2

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        flow_x = np.clip(grid_x + flow[..., 0], 0, w - 1).astype(np.float32)
        flow_y = np.clip(grid_y + flow[..., 1], 0, h - 1).astype(np.float32)

        new_cloud_result = cv2.remap(cloud_prev.astype(np.float32), flow_x, flow_y, cv2.INTER_NEAREST).astype(np.uint8)
    else:
        new_cloud_result = np.zeros_like(cloud_seg_images[0])

    # **7. 计算 avg_euclidean**
    if frame_index % PERIOD == 0 and frame_index > 0:
        start = time.time()

        last_frame_gray = cv2.cvtColor(cloud_seg_images[frame_index], cv2.COLOR_BGR2GRAY)
        flow_period = dis_2.calc(first_frame_gray, last_frame_gray, None)

        # 后面跟你的 Farneback 版本一样
        euclidean_distance = np.sqrt(flow_period[..., 0] ** 2 + flow_period[..., 1] ** 2)
        avg_euclidean = np.mean(euclidean_distance)

        first_frame_gray = last_frame_gray.copy()

        # flow_period = cv2.calcOpticalFlowFarneback(first_frame_gray, last_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # euclidean_distance = np.sqrt(flow_period[..., 0] ** 2 + flow_period[..., 1] ** 2)
        # avg_euclidean = np.mean(euclidean_distance)
        # first_frame_gray = last_frame_gray.copy()
        end = time.time()
        print(end - start)
    selected_frame = edge_seg_images[frame_index].copy() if decision == "Edge" else new_cloud_result.copy()
    selected_text = f"Decision: {decision}"
    color = (0, 255, 0) if decision == "Edge" else (255, 255, 255)

    # **8. 显示所有窗口**
    # cv2.imshow("ECSeg", edge_seg_images[frame_index] if decision == "Edge" else new_cloud_result)
    latency_text = f"Cloud Latency: {average_latency:.0f} ms"
    # cv2.putText(new_cloud_result, latency_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    # cv2.putText(new_cloud_result, f"Avg Euclidean: {avg_euclidean:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #             (255, 255, 255), 4)

    cv2.putText(selected_frame, selected_text, (700, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
    cv2.putText(selected_frame, latency_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(selected_frame, f"Avg Euclidean: {avg_euclidean:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 4)

    # ⭐修改后（复制一份，再写）
    edge_frame_copy = edge_seg_images[frame_index].copy()
    cloud_frame_copy = new_cloud_result.copy()

    cv2.putText(edge_frame_copy, "Edge results", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 4)

    cv2.putText(cloud_frame_copy, "Cloud results + Propagation", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 4)

    # cv2.putText(edge_seg_images[frame_index], "Edge results", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #             (255, 255, 255), 4)
    #
    # cv2.putText(new_cloud_result, "Cloud results + Propagation", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #             (255, 255, 255), 4)


    # cv2.imshow("Raw Data", raw_data_images[frame_index])
    # cv2.imshow("Edge Segmentation", edge_seg_images[frame_index])
    # cv2.imshow("Cloud Segmentation", new_cloud_result)




    separator_vertical = np.ones((512, 10, 3), dtype=np.uint8) * 255  # 竖直分隔线，宽10px，高512px
    separator_horizontal = np.ones((10, 2 * 1024 + 10, 3), dtype=np.uint8) * 255  # 水平分隔线，宽2060px

    # ⭐ Resize四张图（你原本的设定，不变）
    small_raw = cv2.resize(raw_data_images[frame_index], (1024, 512))
    small_edge = cv2.resize(edge_frame_copy, (1024, 512))
    small_cloud = cv2.resize(cloud_frame_copy, (1024, 512))
    small_ecseg = cv2.resize(selected_frame, (1024, 512))

    # ⭐ 拼接（加上分隔线）
    top_row = np.hstack((small_raw, separator_vertical, small_edge))
    bottom_row = np.hstack((small_cloud, separator_vertical, small_ecseg))
    combined_frame = np.vstack((top_row, separator_horizontal, bottom_row))

    # small_raw = cv2.resize(raw_data_images[frame_index], (1024, 512))
    # small_edge = cv2.resize(edge_seg_images[frame_index], (1024, 512))
    # small_cloud = cv2.resize(new_cloud_result, (1024, 512))
    # small_ecseg = cv2.resize(selected_frame, (1024, 512))
    #
    # # **⭐新加：拼接图像⭐**
    # top_row = np.hstack((small_raw, small_edge))
    # bottom_row = np.hstack((small_cloud, small_ecseg))
    # combined_frame = np.vstack((top_row, bottom_row))

    # **⭐新加：只显示一个总窗口⭐**
    cv2.imshow("All Display", combined_frame)

    # **9. 退出检测**
    if cv2.waitKey(1) == 27:
        break

    # **10. 控制帧率**
    time.sleep(max(0, (RAW_EDGE_UPDATE_INTERVAL - (time.time() - loop_start_time) * 1000) / 1000))

cv2.destroyAllWindows()
print("播放结束。")
