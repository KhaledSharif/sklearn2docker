from sklearn.base import ClassifierMixin
from os import path
from shutil import copyfile
from pickle import dump as PickleDumpFile
from tempfile import TemporaryDirectory
from json import dumps

class Sklearn2Docker:
    def __init__(self, classifier, feature_names: list, class_names: list, multi_stage_build=False, production_build=False):
        # assertions to ensure that the given classifier conforms to the `duck-typing` that `scikit-learn` requires
        assert hasattr(classifier, "fit")
        assert hasattr(classifier, "predict")
        assert hasattr(classifier, "predict_proba")

        # type assertions for the feature and class names
        assert isinstance(feature_names, list), type(feature_names)
        assert isinstance(class_names, list), type(class_names)

        self.classifier = classifier
        self.feature_names = feature_names
        self.class_names = class_names

        self.requirements_txt = [
            'flask',
            'flask-cors',
        ]

        if not multi_stage_build:
            self.docker_file = [
                'FROM frolvlad/alpine-python-machinelearning',
                'RUN mkdir /sklearn2docker',
                'COPY ./ /sklearn2docker/',
                'RUN pip install -r /sklearn2docker/requirements.txt',
                'EXPOSE 5000',
                'ENTRYPOINT python /sklearn2docker/api.py',
            ]
        else:
            # TODO: implement Docker multi-stage builds
            raise NotImplementedError()

        if not production_build:
            self.docker_file += [
                'ENTRYPOINT python /sklearn2docker/api.py'
            ]
        else:
            # TODO: alter number of processes depending on cpu count
            self.docker_file += [
                'RUN pip install gunicorn',
                'ENTRYPOINT cd /sklearn2docker && gunicorn -w 4 -b :5000 api:app'
            ]

    def save(self, name="classifier", tag=None):
        from time import time
        import shlex, subprocess

        if tag is None:
            # initialize the tag with the current unix timestamp if not defined
            tag = str(int(time()))

        self.temporary_directory = TemporaryDirectory()

        self.config_file_contents = {
            "feature_names": self.feature_names,
            "class_names": self.class_names,
        }

        self.classifier_path = path.join(self.temporary_directory.name, 'classifier.pkl')
        self.classifier_text_file_path = path.join(self.temporary_directory.name, 'classifier.txt')
        self.classifier_dot_file_path = path.join(self.temporary_directory.name, 'classifier.dot')

        from sklearn.pipeline import Pipeline
        if isinstance(self.classifier, Pipeline):
            from keras import models
            self.docker_file = [
                "FROM tensorflow/tensorflow:1.6.0-rc1-py3",
                "RUN pip install keras"
            ] + self.docker_file[1:]
            self.classifier_nn_model = self.classifier.steps.pop(-1)[-1]
            classifier_model_file_name = 'classifier_nn_model.h5'
            self.classifier_nn_model_path = path.join(self.temporary_directory.name, classifier_model_file_name)
            models.save_model(self.classifier_nn_model.model, self.classifier_nn_model_path)
            self.config_file_contents["keras_model_weights"] = "/sklearn2docker/" + classifier_model_file_name

        with open(self.classifier_path, 'wb') as f:
            PickleDumpFile(self.classifier, f)

        try:
            # get the `export_graphviz` output as a string
            from sklearn.tree import export_graphviz
            export_graphviz_string = export_graphviz(
                self.classifier,
                out_file=None,
                feature_names=self.feature_names,
                class_names=self.class_names,
                label='none',
                impurity=False,
                rounded=True,
                filled=True,
                proportion=True,
            )
            # write the dot file
            with open(self.classifier_dot_file_path, 'w') as f:
                f.write(export_graphviz_string)
        except:
            self.classifier_dot_file_path = ""
            pass

        # write the feature names text file
        with open(self.classifier_text_file_path, 'w') as f:
            f.write("\n".join(self.feature_names) + "\n")

        self.docker_file_path = path.join(self.temporary_directory.name, 'Dockerfile')
        self.pip_requirements_file_path = path.join(self.temporary_directory.name, 'requirements.txt')
        self.json_config_file_path = path.join(self.temporary_directory.name, 'config.json')

        # write Dockerfile
        with open(self.docker_file_path, 'w') as f:
            f.write("\n".join(self.docker_file))

        # write pip requirements text file
        with open(self.pip_requirements_file_path, 'w') as f:
            f.write("\n".join(self.requirements_txt))

        # write the configuration JSON file
        with open(self.json_config_file_path, 'w') as f:
            f.write(dumps(
                self.config_file_contents
            ))

        # copy over the api.py file
        api_file_path = path.join(path.abspath(path.dirname(__file__)), "api.py")
        copyfile(api_file_path, path.join(self.temporary_directory.name, "api.py"))

        args = shlex.split("docker build --file {} --tag {}:{} {}".format(
            self.docker_file_path,
            name,
            tag,
            self.temporary_directory.name,
        ))

        print("Now attempting to run the command: [{}]".format(" ".join(args)))
        process = subprocess.Popen(args, stdout=subprocess.PIPE)

        print("=" * 80)

        while process.poll() is None:
            output = process.stdout.readline()
            if output:
                print(">", output.decode().strip())

        print("=" * 80)

        rc = process.poll()

        assert rc == 0 or rc is None, \
            "Error! The Docker build command failed with return code {}.".format(rc)

        print("Success! You can now run your Docker container using the following command:")
        print("\t docker run -d -p 5000:5000 {}:{}".format(name, tag))
