import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from threading import Thread

class YoloRealSenseNode(Node):
    def __init__(self):
        super().__init__('yolo_realsense_node')
        
        self.declare_parameter('model_path', '/home/jetson/ros2/ros2_ws/models/best.pt')
        self.declare_parameter('frame_width', 640)  # 分辨率宽度
        self.declare_parameter('frame_height', 480)  # 分辨率高度
        self.declare_parameter('fps', 15)  # 帧率

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.frame_width = self.get_parameter('frame_width').get_parameter_value().integer_value
        self.frame_height = self.get_parameter('frame_height').get_parameter_value().integer_value
        self.fps = self.get_parameter('fps').get_parameter_value().integer_value

        self.bridge = CvBridge()

        # 加载 YOLO 模型
        self.model = YOLO(self.model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        if self.device == 'cuda':
            self.model.half()

        # 初始化 RealSense 管道
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.frame_width, self.frame_height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.frame_width, self.frame_height, rs.format.z16, self.fps)
        
        try:
            self.pipeline.start(self.config)
        except Exception as e:
            self.get_logger().error(f"Error starting RealSense pipeline: {e}")
            raise

        self.align = rs.align(rs.stream.color)
        self.publisher = self.create_publisher(Image, 'yolo/detections', 10)

        # 开启图像处理线程
        self.processing_thread = Thread(target=self.process_images, daemon=True)
        self.processing_thread.start()

    def calculate_angle(self, vector):
        # 计算与法向量 (0,0,1) 的夹角
        normal_vector = np.array([0, 0, 1], dtype=np.float32)
        vector = np.array(vector, dtype=np.float32)
        
        dot_product = np.dot(vector, normal_vector)
        norm_vector = np.linalg.norm(vector)
        norm_normal = np.linalg.norm(normal_vector)
        
        cos_theta = dot_product / (norm_vector * norm_normal + 1e-8)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def calculate_real_length(self, pixel_length, depth, intrinsics):
        """根据像素长度、深度和相机内参计算实际长度（单位：米）"""
        fx = intrinsics.fx  # 焦距（像素）
        return pixel_length * depth / fx

    def process_images(self):
        while rclpy.ok():
            try:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    self.get_logger().warning("No color or depth frame received.")
                    continue

                frame = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # 获取深度内参用于尺寸计算
                depth_intri = depth_frame.profile.as_video_stream_profile().intrinsics

                # 使用 YOLO 模型进行目标检测
                results = self.model(frame)

                if results:
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # 提取边界框坐标、类别与置信度
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                            label = result.names[box.cls[0].item()]
                            confidence = box.conf[0].item()

                            # 计算边界框中心
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # 获取中心处深度（单位：米）
                            object_distance = depth_frame.get_distance(center_x, center_y)

                            # 绘制 YOLO 边界框及标签
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            # 将中心点转换到相机坐标系（单位：mm）
                            dis = depth_frame.get_distance(center_x, center_y)
                            camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intri, (center_x, center_y), dis)
                            camera_xyz = np.round(np.array(camera_xyz)*100, 2)
                            cv2.circle(frame, (center_x, center_y), 2, (255, 255, 255), 2)
                            cv2.putText(frame, f"({camera_xyz[0]:.2f}, {camera_xyz[1]:.2f}, {camera_xyz[2]:.2f})", 
                                        (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(frame, f"Distance: {object_distance:.3f}m", (x1, y2 + 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            # 计算目标与 Z 轴（法向量 0,0,1）的夹角
                            angle_deg = self.calculate_angle(camera_xyz)
                            cv2.putText(frame, f"Angle: {angle_deg:.2f}", (x1, y2 + 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            # 在检测区域内部进行轮廓分析以获得实际尺寸信息
                            roi = frame[y1:y2, x1:x2]
                            if roi.size > 0:
                                # 灰度化并采用 Otsu 阈值法
                                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                # 形态学操作去噪
                                kernel = np.ones((5, 5), np.uint8)
                                dilated = cv2.dilate(thresh, kernel, iterations=1)
                                eroded = cv2.erode(dilated, kernel, iterations=1)
                                # 查找轮廓
                                contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                for contour in contours:
                                    area = cv2.contourArea(contour)
                                    if area > 500:  # 过滤小区域
                                        M = cv2.moments(contour)
                                        if M["m00"] == 0:
                                            continue
                                        # 计算轮廓在 ROI 坐标系中的质心
                                        cx_roi = int(M["m10"] / M["m00"])
                                        cy_roi = int(M["m01"] / M["m00"])
                                        # 转换为全图坐标
                                        cx = x1 + cx_roi
                                        cy = y1 + cy_roi

                                        # 利用 PCA 分析轮廓主方向
                                        data_points = np.array(contour).reshape(-1, 2).astype(np.float32)
                                        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_points, mean=np.empty((0), dtype=np.float32))
                                        major_vector = eigenvectors[0]
                                        major_length = 2 * np.sqrt(eigenvalues[0][0])
                                        minor_length = 2 * np.sqrt(eigenvalues[1][0])
                                        
                                        # 获取当前点深度并计算实际尺寸（转换为厘米）
                                        depth_val = depth_frame.get_distance(cx, cy)
                                        if depth_val <= 0:
                                            continue
                                        major_real = self.calculate_real_length(major_length, depth_val, depth_intri) * 100
                                        minor_real = self.calculate_real_length(minor_length, depth_val, depth_intri) * 100

                                        # 在图像上绘制主轴线
                                        pt1 = (int(cx - major_vector[0] * major_length / 2), int(cy - major_vector[1] * major_length / 2))
                                        pt2 = (int(cx + major_vector[0] * major_length / 2), int(cy + major_vector[1] * major_length / 2))
                                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                                        # 标注尺寸信息
                                        size_text = f"L:{major_real:.2f}cm, S:{minor_real:.2f}cm"
                                        cv2.putText(frame, size_text, (cx, cy - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # 标注中心坐标
                            center_text = f"Center: ({center_x}, {center_y})"
                            cv2.putText(frame, center_text, (x1, y2 + 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # 发布 ROS2 图像消息
                self.publisher.publish(self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))

                # 显示处理后图像
                cv2.imshow('YOLO RealSense', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                self.get_logger().error(f"Error processing image: {e}")

        cv2.destroyAllWindows()

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception as e:
            self.get_logger().error(f"Error stopping RealSense pipeline: {e}")
        finally:
            cv2.destroyAllWindows()
            super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloRealSenseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
