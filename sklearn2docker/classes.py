import pickle

from os.path import isfile


class BaseClassifier:
    def __init__(self):
        pass

    def predict(self, data):
        raise NotImplementedError()

    def predict_proba(self, data):
        raise NotImplementedError()

class ScikitLearnClassifier(BaseClassifier):
    def __init__(self, pickle_file: str):
        super().__init__()
        self.classifier_object = pickle.load(pickle_file)

    def predict(self, data):
        return self.classifier_object.predict(data)

    def predict_proba(self, data):
        return self.classifier_object.predict_proba(data)

class KerasBinaryClassifier(BaseClassifier):
    def __init__(self, pickle_file: str, keras_model_weights: str):
        super().__init__()
        global tensorflow_default_graph
        with open(pickle_file, "rb") as c:
            self.classifier_object = pickle.load(c)
        from keras import models
        import tensorflow as tf
        self.classifier_object.steps.append(('keras_neural_network', models.load_model(keras_model_weights)))
        tensorflow_default_graph = tf.get_default_graph()

    def predict(self, data):
        global tensorflow_default_graph
        with tensorflow_default_graph.as_default():
            prediction = self.classifier_object.predict(data.values)[:, 0].tolist()
            prediction = [1 if x > 0.5 else 0 for x in prediction]
            return prediction

    def predict(self, data):
        global tensorflow_default_graph
        with tensorflow_default_graph.as_default():
            prediction = self.classifier_object.predict(data.values)[:, 0].tolist()
            prediction = [1 if x > 0.5 else 0 for x in prediction]
            return prediction

class Classifier:
    def __init__(self, configuration_file: dict):
        self.expected_column_names = configuration_file["feature_names"]
        self.class_names = configuration_file["class_names"]

        assert isfile("/sklearn2docker/classifier.pkl")

        if "keras_model_weights" in configuration_file:
            assert isfile(configuration_file["keras_model_weights"])

            if len(self.class_names) == 2:
                self.classifier_object = KerasBinaryClassifier("/sklearn2docker/classifier.pkl", configuration_file["keras_model_weights"])
            else:
                raise NotImplementedError()
        else:
            self.classifier_object = ScikitLearnClassifier("/sklearn2docker/classifier.pkl")

        assert hasattr(self.classifier_object, "predict")
        assert hasattr(self.classifier_object, "predict_proba")

    def predict(self, values):
        return self.classifier_object.predict(values)

    def predict_proba(self, values):
        return self.classifier_object.predict_proba(values)