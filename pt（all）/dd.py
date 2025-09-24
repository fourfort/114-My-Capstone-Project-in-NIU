import csv
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import os

# 加载两个不同的模型
model_yolov8s = YOLO(r'D:\python main\123\123main\pt（all）\best(yolov8s).pt')


model = YOLO(r'D:\python main\123\123main\pt（all）\best（CSPshufflenet+C2f）.pt')

# 直接打印模型的結構
print(model)