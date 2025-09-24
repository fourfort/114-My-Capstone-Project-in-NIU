# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 用來存放最新的模擬資料（你也可以存進 DB）
latest_data = {
    "gradeA": 3,
    "gradeB1": 34,
    "gradeB2": 233,
    "sizeL": 45,
    "sizeM": 234,
    "sizeS": 23
}

@app.route("/upload", methods=["POST"])
def upload():
    global latest_data
    data = request.get_json()
    if data:
        latest_data.update(data)
        return jsonify({"status": "success", "msg": "資料已更新"}), 200
    return jsonify({"status": "fail", "msg": "未收到資料"}), 400

@app.route("/data", methods=["GET"])
def get_data():
    return jsonify(latest_data)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
