import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 用來存放最新的模擬資料（你也可以存進 DB）
latest_data = {
    "gradeA": 0,
    "gradeB1": 0,
    "gradeB2": 0,
    "sizeL": 0,
    "sizeM": 0,
    "sizeS": 0
}


@app.route("/upload", methods=["POST"])
def upload():
    global latest_data
    data = request.get_json()

    if data:
        # 记录更新前的 latest_data 状态
        logging.info(f"更新前的資料: {latest_data}")

        # 更新数据中相应的标记
        if "gradeA" in data :
            latest_data["gradeA"] += 1
        if "gradeB1" in data :
            latest_data["gradeB1"] += 1
        if "gradeB2" in data :
            latest_data["gradeB2"] += 1
        
        # 根据需要更新大小尺寸数量
        if "sizeL" in data:
            latest_data["sizeL"] += data["sizeL"]
        if "sizeM" in data:
            latest_data["sizeM"] += data["sizeM"]
        if "sizeS" in data:
            latest_data["sizeS"] += data["sizeS"]

        # 记录更新后的 latest_data 状态
        logging.info(f"更新後的資料: {latest_data}")

        return jsonify({"status": "success", "msg": "資料已更新"}), 200
    
    return jsonify({"status": "fail", "msg": "未收到資料"}), 400

@app.route("/data", methods=["GET"])
def get_data():
    return jsonify(latest_data)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
