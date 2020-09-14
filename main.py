from flask import Flask, json, request
from flask_cors import CORS, cross_origin
from train import train_model
import numpy as np

api = Flask(__name__)
CORS(api)
model = train_model()


@api.route('/evaluate', methods=['POST'])
@cross_origin()
def evaluate_data():
    prediction = model.predict(np.array(request.json['values']))
    return json.dumps({"result": prediction.tolist()})


@api.route('/', methods=['GET'])
@cross_origin()
def get_route():
    return 'Hello predict eating model'


if __name__ == '__main__':
    api.run()
