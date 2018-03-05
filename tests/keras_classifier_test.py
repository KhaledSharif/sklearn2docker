from tests.base_unit_test import BaseUnitTest

class KerasClassifierUnitTest(BaseUnitTest):
    @staticmethod
    def create_binary_classification_model():
        from keras.models import Sequential
        from keras.layers import Dense, Dropout

        model = Sequential()
        model.add(Dense(64, input_shape=(30,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    @staticmethod
    def create_categorical_classification_model():
        from keras.models import Sequential
        from keras.layers import Dense, Dropout

        model = Sequential()
        model.add(Dense(64, input_shape=(30,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def test_binary_classifier(self):
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.datasets import load_breast_cancer
        from sklearn import preprocessing
        from pandas import DataFrame
        from numpy import array
        from os import system
        from pandas import read_json
        from requests import post

        breast_cancer = load_breast_cancer()
        input_df = DataFrame(data=breast_cancer['data'], columns=breast_cancer['feature_names'])
        model = Pipeline([
            ('rescale', preprocessing.StandardScaler()),
            ('min_max', preprocessing.MinMaxScaler((-1, 1,))),
            ('nn', KerasClassifier(build_fn=self.create_binary_classification_model, epochs=1, verbose=1)),
        ])
        X, Y = input_df.values, array(breast_cancer['target'])
        model.fit(X, Y)

        # convert classifier to Docker container
        from sklearn2docker.constructor import Sklearn2Docker
        s2d = Sklearn2Docker(
            classifier=model,
            feature_names=list(input_df),
            class_names=breast_cancer['target_names'].tolist(),
        )
        s2d.save(
            name="classifier",
            tag="keras",
        )

        # # run your Docker container as a detached process
        system("docker run -d -p {}:5000 --name {} classifier:keras && sleep 5".format(self.port, self.container_name))

        # send your training data as a json string
        request = post("http://localhost:{}/predict/split".format(self.port), json=input_df.to_json(orient="split"))
        result = read_json(request.content.decode(), orient="split")
        self.assertEqual(len(list(result)), 1)
        self.assertEqual(len(result), len(input_df))

        request = post("http://localhost:{}/predict_proba/split".format(self.port), json=input_df.to_json(orient="split"))
        result = read_json(request.content.decode(), orient="split")
        self.assertEqual(len(list(result)), 1)
        self.assertEqual(len(result), len(input_df))

    def test_categorical_classifier(self):
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.datasets import load_iris
        from sklearn import preprocessing
        from pandas import DataFrame
        from numpy import array
        from os import system
        from pandas import read_json
        from requests import post

        iris = load_iris()
        input_df = DataFrame(data=iris['data'], columns=iris['feature_names'])
        model = Pipeline([
            ('rescale', preprocessing.StandardScaler()),
            ('min_max', preprocessing.MinMaxScaler((-1, 1,))),
            ('nn', KerasClassifier(build_fn=self.create_categorical_classification_model, epochs=1, verbose=1)),
        ])
        X, Y = input_df.values, array(iris['target'])
        model.fit(X, Y)

        # convert classifier to Docker container
        from sklearn2docker.constructor import Sklearn2Docker
        s2d = Sklearn2Docker(
            classifier=model,
            feature_names=list(input_df),
            class_names=iris['target_names'].tolist(),
        )
        s2d.save(
            name="classifier",
            tag="keras",
        )

        # # run your Docker container as a detached process
        system("docker run -d -p {}:5000 --name {} classifier:keras && sleep 5".format(self.port, self.container_name))

        # send your training data as a json string
        request = post("http://localhost:{}/predict/split".format(self.port), json=input_df.to_json(orient="split"))
        result = read_json(request.content.decode(), orient="split")
        self.assertEqual(len(list(result)), 1)
        self.assertEqual(len(result), len(input_df))

        request = post("http://localhost:{}/predict_proba/split".format(self.port), json=input_df.to_json(orient="split"))
        result = read_json(request.content.decode(), orient="split")
        self.assertEqual(len(list(result)), 3)
        self.assertEqual(len(result), len(input_df))


    def test_barebones_keras(self):
        from sklearn.datasets import load_iris
        from pandas import DataFrame
        from numpy import array
        from os import system
        from pandas import read_json
        from requests import post

        iris = load_iris()
        input_df = DataFrame(data=iris['data'], columns=iris['feature_names'])
        model = self.create_categorical_classification_model()
        X, Y = input_df.values, array(iris['target'])
        model.fit(X, Y)

        # convert classifier to Docker container
        from sklearn2docker.constructor import Sklearn2Docker
        s2d = Sklearn2Docker(
            classifier=model,
            feature_names=list(input_df),
            class_names=iris['target_names'].tolist(),
        )
        s2d.save(
            name="classifier",
            tag="keras",
        )

        # # run your Docker container as a detached process
        system("docker run -d -p {}:5000 --name {} classifier:keras && sleep 5".format(self.port, self.container_name))

        # send your training data as a json string
        request = post("http://localhost:{}/predict/split".format(self.port), json=input_df.to_json(orient="split"))
        result = read_json(request.content.decode(), orient="split")
        self.assertEqual(len(list(result)), 1)
        self.assertEqual(len(result), len(input_df))

        request = post("http://localhost:{}/predict_proba/split".format(self.port), json=input_df.to_json(orient="split"))
        result = read_json(request.content.decode(), orient="split")
        self.assertEqual(len(list(result)), 3)
        self.assertEqual(len(result), len(input_df))


if __name__ == '__main__':
    from unittest import main
    main()
