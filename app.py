from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import utils

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_ORIGINS'] = ['http://localhost:5173']
app.config['Access-Control-Allow-Origin'] = '*'

@app.route('/')
@cross_origin()
def hello_world():
    return 'Hello, World! Server is running...';

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    print("Request received")
    data = request.json.get('data')
    size = request.json.get('size')
    print(data)
    response = utils.make_prediction(data,int(size))
    print(response)
    return jsonify(response)


if __name__ == '__main__':
    print("Starting Python Flask Server For  Prediction...")
    app.run(port=8080)