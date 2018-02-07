# create Sklearn classifier
from pandas import DataFrame
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
s2d.save()