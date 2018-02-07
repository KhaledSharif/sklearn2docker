from json import loads
from flask import Flask, request
from flask_cors import CORS
from pandas import read_json
from os import environ

app = Flask(__name__, template_folder='.')
CORS(app)


def get_environment_variable_as_list(name: str) -> list:
    return environ.get(name).split(",")


classifier = environ.get("CLASSIFIER_NAME")
classifier_expected_column_names = get_environment_variable_as_list("CLASSIFIER_EXPECTED_COLUMN_NAMES")
classifier_class_names = get_environment_variable_as_list("CLASSIFIER_CLASS_NAMES")


def perform_prediction(probabilistic=False) -> str:
    # attempt to retrieve orient as string, else default to `table`
    orient = request.args.get('orient', 'table', type=str)

    # attempt to retrieve data as json string, else fail
    data = request.args.get('data', type=str)

    # convert data to json
    data = loads(data)

    # convert json to pandas dataframe
    data = read_json(data, orient=orient)

    # convert dataframe back to json string
    data = data.to_json(orient=orient)

    return data


def return_dot_file():
    pass


@app.route('/predict')
def get_predictions():
    return perform_prediction()


@app.route('/predict_proba')
def get_predicted_probabilities():
    return perform_prediction(probabilistic=True)


@app.route('/dot')
def get_dot_file():
    return return_dot_file()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
