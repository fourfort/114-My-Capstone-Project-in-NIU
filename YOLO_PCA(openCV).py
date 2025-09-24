import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# ----------------------------
# 初始化 RealSense 相机
# ----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 获取深度传感器内参和深度比例因子
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # 深度单位比例因子
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# ----------------------------
# 加载 YOLO 模型
# ----------------------------
# 请替换为你模型的实际路径
model = YOLO("/home/jetson/ros2/ros2_ws/models/best.pt")

def calculate_real_length(pixel_length, depth, intrinsics):
    """根据像素长度和深度计算实际长度（单位：米）"""
    fx = intrinsics.fx  # 焦距（像素）
    return pixel_length * depth / fx

try:
    while True:
        # 获取图像帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # 转换为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())
        
        # ----------------------------
        # YOLO 目标检测
        # ----------------------------
        results = model(color_image)
        # 遍历检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标、标签和置信度
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                label = result.names[box.cls[0].item()]
                confidence = box.conf[0].item()

                # 在原图上绘制 YOLO 边界框与标签
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(color_image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # ----------------------------
                # 对边界框内的区域进行轮廓分析计算尺寸
                # ----------------------------
                # 裁剪出目标区域
                roi = color_image[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # 转换为灰度图像并进行阈值化处理（Otsu方法）
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # 形态学操作去除噪声
                kernel = np.ones((5, 5), np.uint8)
                dilated = cv2.dilate(thresh, kernel, iterations=1)
                eroded = cv2.erode(dilated, kernel, iterations=1)
                # 查找轮廓
                contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # 过滤较小轮廓
                        # 计算轮廓的质心（在 ROI 坐标系中）
                        M = cv2.moments(contour)
                        if M["m00"] == 0:
                            continue
                        cx_roi = int(M["m10"] / M["m00"])
                        cy_roi = int(M["m01"] / M["m00"])
                        # 转换 ROI 中的坐标到原图坐标
                        cx = x1 + cx_roi
                        cy = y1 + cy_roi

                        # 利用 PCA 分析轮廓主方向
                        data_points = np.array(contour).reshape(-1, 2).astype(np.float32)
                        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_points, mean=np.empty((0), dtype=np.float32))
                        major_vector = eigenvectors[0]
                        major_length = 2 * np.sqrt(eigenvalues[0][0])
                        minor_length = 2 * np.sqrt(eigenvalues[1][0])

                        # 获取轮廓中心点处的深度值
                        depth = depth_frame.get_distance(cx, cy)
                        if depth <= 0:
                            continue

                        # 将像素长度转换为实际长度（单位：米），并转换为厘米显示
                        major_real = calculate_real_length(major_length, depth, intrinsics) * 100
                        minor_real = calculate_real_length(minor_length, depth, intrinsics) * 100

                        # 在原图上绘制主轴方向
                        pt1 = (int(cx - major_vector[0] * major_length / 2), int(cy - major_vector[1] * major_length / 2))
                        pt2 = (int(cx + major_vector[0] * major_length / 2), int(cy + major_vector[1] * major_length / 2))
                        cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)

                        # 标注尺寸与深度信息
                        text = f"L:{major_real:.2f}cm, S:{minor_real:.2f}cm, D:{depth:.2f}m"
                        cv2.putText(color_image, text, (cx, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示结果图像
        cv2.imshow("Combined Detection", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
