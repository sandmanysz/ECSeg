import numpy as np
import cv2
import time
from stable_baselines3 import PPO

# === 初始化数据 ===
x_data, latency_data, euclidean_data = [], [], []
plot_width, plot_height = 640, 480
margin = 50

# ✅ 固定纵坐标范围
max_latency = 4000
max_euclidean = 80

def draw_single_plot(value_list, color, label_text, max_value, max_points=100):
    canvas = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255

    # 画坐标轴
    cv2.line(canvas, (margin, margin), (margin, plot_height - margin), (0, 0, 0), 2)
    cv2.line(canvas, (margin, plot_height - margin), (plot_width - margin, plot_height - margin), (0, 0, 0), 2)

    if len(value_list) < 2:
        return canvas

    values = value_list[-max_points:]
    x_vals = list(range(len(values)))

    for i in range(1, len(x_vals)):
        x1 = int(margin + (x_vals[i-1] / max_points) * (plot_width - 2*margin))
        x2 = int(margin + (x_vals[i] / max_points) * (plot_width - 2*margin))

        y1 = int(plot_height - margin - (values[i-1] / max_value) * (plot_height - 2*margin))
        y2 = int(plot_height - margin - (values[i] / max_value) * (plot_height - 2*margin))

        cv2.line(canvas, (x1, y1), (x2, y2), color, 2)

    cv2.putText(canvas, f'{label_text} (max={max_value})', (margin, margin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return canvas

# === 你的原始代码 ===
ppo_agent = PPO.load("ppo_checkpoint/ppo_offloading_6400_3.zip", device="cpu")

RAW_EDGE_UPDATE_INTERVAL = 60
PERIOD = 17

raw_data_images = np.load("raw_data_half.npy")
edge_seg_images = np.load("edge_segmentation_results_half.npy")
cloud_seg_images = np.load("cloud_segmentation_results_half.npy")
image_latency = np.load("latency_sample_v2.npy")

min_frames = min(raw_data_images.shape[0], edge_seg_images.shape[0], cloud_seg_images.shape[0])

frame_index = 0
latency_index = 0
current_latency = image_latency[0] if len(image_latency) > 0 else 100
avg_euclidean = 0.0
latency_buffer = []
average_latency = current_latency

cloud_queue = []
latest_cloud_frame = -1
current_cloud_task = None

dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
dis_2 = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

cv2.namedWindow("All Display", cv2.WINDOW_NORMAL)
cv2.resizeWindow("All Display", 2048, 1024)

cv2.namedWindow("Latency Plot", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Latency Plot", plot_width, plot_height)

cv2.namedWindow("Euclidean Plot", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Euclidean Plot", plot_width, plot_height)

print("开始模拟视频播放（按 ESC 退出）。")

start_time = time.time()
first_frame_gray = cv2.cvtColor(raw_data_images[0], cv2.COLOR_BGR2GRAY)

while True:
    loop_start_time = time.time()
    elapsed_time = (loop_start_time - start_time) * 1000

    if frame_index % PERIOD == 0 and frame_index > 0:
        average_latency = np.mean(latency_buffer) if latency_buffer else current_latency
        latency_buffer = []

    if frame_index < PERIOD:
        decision = "Edge"
    elif frame_index % PERIOD == 0:
        print(f"Avg Euclidean: {avg_euclidean}, Avg Latency: {average_latency}")
        observation = np.array([avg_euclidean/300, (average_latency-22)/2961]).reshape(1, -1)
        action, _ = ppo_agent.predict(observation, deterministic=True)
        decision = "Edge" if action == 0 else "Cloud"

    if elapsed_time >= (frame_index + 1) * RAW_EDGE_UPDATE_INTERVAL:
        frame_index += 1
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
            start_time = time.time()

            # 清空 plot 数据
            x_data.clear()
            latency_data.clear()
            euclidean_data.clear()

            continue

        if latency_index < len(image_latency):
            current_latency = image_latency[latency_index]
            latency_index += 1
        else:
            current_latency = image_latency[-1]

        latency_buffer.append(current_latency)
        cloud_queue.append((frame_index, elapsed_time + current_latency))

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

    if frame_index % PERIOD == 0 and frame_index > 0:
        start = time.time()

        last_frame_gray = cv2.cvtColor(cloud_seg_images[frame_index], cv2.COLOR_BGR2GRAY)
        flow_period = dis_2.calc(first_frame_gray, last_frame_gray, None)

        euclidean_distance = np.sqrt(flow_period[..., 0] ** 2 + flow_period[..., 1] ** 2)
        avg_euclidean = np.mean(euclidean_distance)

        first_frame_gray = last_frame_gray.copy()

        end = time.time()
        print(end - start)

        # 更新 plot 数据
        x_data.append(frame_index)
        latency_data.append(average_latency)
        euclidean_data.append(avg_euclidean)

    # === 实时绘图 ===
    latency_plot = draw_single_plot(latency_data, (255, 0, 0), 'Latency', max_latency)
    euclidean_plot = draw_single_plot(euclidean_data, (0, 0, 255), 'Euclidean', max_euclidean)

    cv2.imshow("Latency Plot", latency_plot)
    cv2.imshow("Euclidean Plot", euclidean_plot)

    selected_frame = edge_seg_images[frame_index].copy() if decision == "Edge" else new_cloud_result.copy()
    selected_text = f"Decision: {decision}"
    color = (0, 255, 0) if decision == "Edge" else (255, 255, 255)

    latency_text = f"Cloud Latency: {average_latency:.0f} ms"

    cv2.putText(selected_frame, selected_text, (700, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
    cv2.putText(selected_frame, latency_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(selected_frame, f"Avg Euclidean: {avg_euclidean:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 4)

    edge_frame_copy = edge_seg_images[frame_index].copy()
    cloud_frame_copy = new_cloud_result.copy()

    cv2.putText(edge_frame_copy, "Edge results", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 4)

    cv2.putText(cloud_frame_copy, "Cloud results + Propagation", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 4)

    separator_vertical = np.ones((512, 10, 3), dtype=np.uint8) * 255
    separator_horizontal = np.ones((10, 2 * 1024 + 10, 3), dtype=np.uint8) * 255

    small_raw = cv2.resize(raw_data_images[frame_index], (1024, 512))
    small_edge = cv2.resize(edge_frame_copy, (1024, 512))
    small_cloud = cv2.resize(cloud_frame_copy, (1024, 512))
    small_ecseg = cv2.resize(selected_frame, (1024, 512))

    top_row = np.hstack((small_raw, separator_vertical, small_edge))
    bottom_row = np.hstack((small_cloud, separator_vertical, small_ecseg))
    combined_frame = np.vstack((top_row, separator_horizontal, bottom_row))

    cv2.imshow("All Display", combined_frame)

    if cv2.waitKey(1) == 27:
        break

    time.sleep(max(0, (RAW_EDGE_UPDATE_INTERVAL - (time.time() - loop_start_time) * 1000) / 1000))

cv2.destroyAllWindows()
print("播放结束。")
