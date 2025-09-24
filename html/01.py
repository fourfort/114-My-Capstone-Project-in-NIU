# simulate_data.py
import requests
import random
import time

def generate_fake_data():
    return {
        "gradeA": random.randint(80, 100),
        "gradeB1": random.randint(10, 20),
        "gradeB2": random.randint(0, 10),
        "sizeL": random.randint(40, 60),
        "sizeM": random.randint(30, 50),
        "sizeS": random.randint(10, 30)
    }

while True:
    data = generate_fake_data()
    res = requests.post("http://localhost:5000/upload", json=data)
    print("已送出：", data)
    time.sleep(10)  # 每 10 秒送一次
