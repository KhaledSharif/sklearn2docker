from flask import Flask, request
from flask_cors import CORS
from pandas import read_json
from json import load
from pandas import DataFrame
from sklearn.tree import export_graphviz
from sklearn2docker.classes import *

tensorflow_default_graph = None

app = Flask(__name__)
CORS(app)

with open('/sklearn2docker/config.json') as cf:
    classifier = Classifier(load(cf))

# the recommended pandas `orient` is `split`
# see: https://github.com/pandas-dev/pandas/issues/18912
def perform_prediction(probabilistic, orient) -> str:
    global tensorflow_default_graph
    global classifier

    # attempt to retrieve data as json string, else fail
    data = request.get_json(force=True)

    # convert json to pandas dataframe
    data = read_json(data, orient=orient)

    # reorder dataframe with our expected column names
    data = data[classifier.expected_column_names]

    # perform prediction
    if not probabilistic:
        if tensorflow_default_graph:
            with tensorflow_default_graph.as_default():
                prediction = classifier.classifier_object.predict(data.values)[:, 0].tolist()
                prediction = [1 if x > 0.5 else 0 for x in prediction]
        else:
            prediction = classifier.classifier_object.predict(data.values).tolist()

        prediction = [classifier.class_names[x] for x in prediction]
        prediction_dataframe = DataFrame()
        prediction_dataframe["prediction"] = prediction
    else:
        if tensorflow_default_graph:
            with tensorflow_default_graph.as_default():
                prediction = classifier.classifier_object.predict_proba(data.values)[:, 0]

                print(prediction)
        else:
            prediction = classifier.classifier_object.predict_proba(data.values).tolist()

        prediction_dataframe = DataFrame(data=prediction, columns=classifier.class_names)

    # set the correct index
    prediction_dataframe.index = data.index

    # convert dataframe back to json string
    data = prediction_dataframe.to_json(orient=orient)

    return data


def return_dot_file():
    return export_graphviz(classifier.classifier_object, out_file=None)


@app.route('/<prediction_type>/<pandas_orient>', methods=['POST'])
def get_predictions(prediction_type: str, pandas_orient: str):
    assert prediction_type in ["predict", "predict_proba"]
    return perform_prediction(prediction_type == "predict_proba", pandas_orient)


@app.route('/dot', methods=['GET'])
def get_dot_file():
    return return_dot_file()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
