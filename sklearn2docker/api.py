from flask import Flask, request
from flask_cors import CORS
from pandas import read_json
from json import load
import pickle
from pandas import DataFrame
from sklearn.tree import export_graphviz

app = Flask(__name__, template_folder='/code/')
CORS(app)

with open('/code/config.json') as cf:
    configuration_file = load(cf)

classifier_expected_column_names = configuration_file["feature_names"]
classifier_class_names = configuration_file["class_names"]

with open("/code/classifier.pkl", "rb") as c:
    classifier_object = pickle.load(c)


# the recommended pandas `orient` is `split`
# see: https://github.com/pandas-dev/pandas/issues/18912
def perform_prediction(probabilistic, orient) -> str:
    # attempt to retrieve data as json string, else fail
    data = request.get_json(force=True)

    # convert json to pandas dataframe
    data = read_json(data, orient=orient)

    # reorder dataframe with our expected column names
    data = data[classifier_expected_column_names]

    # perform prediction
    if not probabilistic:
        prediction = classifier_object.predict(data.values).tolist()
        prediction = [classifier_class_names[x] for x in prediction]
        prediction_dataframe = DataFrame()
        prediction_dataframe["prediction"] = prediction
    else:
        prediction = classifier_object.predict_proba(data.values)
        prediction_dataframe = DataFrame(data=prediction, columns=classifier_class_names)

    # set the correct index
    prediction_dataframe.index = data.index

    # convert dataframe back to json string
    data = prediction_dataframe.to_json(orient=orient)

    return data


def return_dot_file():
    return export_graphviz(classifier_object, out_file=None)


@app.route('/<prediction_type>/<pandas_orient>', methods=['POST'])
def get_predictions(prediction_type: str, pandas_orient: str):
    assert prediction_type in ["predict", "predict_proba"]
    return perform_prediction(prediction_type == "predict_proba", pandas_orient)


@app.route('/dot', methods=['GET'])
def get_dot_file():
    return return_dot_file()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
