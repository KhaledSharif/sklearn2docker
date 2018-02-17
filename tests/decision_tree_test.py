# create Sklearn classifier
from pandas import DataFrame, read_json
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
input_df = DataFrame(data=iris['data'], columns=iris['feature_names'])
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(input_df.values, iris['target'])

# convert classifier to Docker container
from sklearn2docker.constructor import Sklearn2Docker
s2d = Sklearn2Docker(
    classifier=clf,
    feature_names=iris['feature_names'],
    class_names=iris['target_names'].tolist(),
)
s2d.save(
    name="classifier",
    tag="iris",
)

# run your Docker container as a detached process
from os import system
system("docker run -d -p 5000:5000 classifier:iris && sleep 5")

# send your training data as a json string
from requests import post
request = post("http://localhost:5000/predict/split", json=input_df.to_json(orient="split"))
result = read_json(request.content.decode(), orient="split")
print(result.head())

request = post("http://localhost:5000/predict_proba/split", json=input_df.to_json(orient="split"))
result = read_json(request.content.decode(), orient="split")
print(result.head())
