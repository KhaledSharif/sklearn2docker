# sklearn2docker
#### Convert your trained scikit-learn classifier to a Docker container with a pre-configured API

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](http://www.gnu.org/licenses/lgpl-3.0)

## Installation

The easiest way to install `sklearn2docker` with all its dependencies is through `pip`:

```bash
pip install git+git://github.com/KhaledSharif/sklearn2docker.git
```

## Getting started

First, create your `sklearn` classifier. In this example we will use the [Iris dataset](http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).

```python
from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
input_df = DataFrame(data=iris['data'], columns=iris['feature_names'])
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(input_df.values, iris['target'])
```

Second, import the `Sklearn2Docker` class and use it to build your container.

```python
from sklearn2docker.constructor import Sklearn2Docker

s2d = Sklearn2Docker(
    classifier=clf,
    feature_names=iris['feature_names'],
    class_names=iris['target_names'].tolist()
)
s2d.save(name="classifier", tag="iris")
```

The name and tag arguments we passed to the `save` function are the name and tag of the Docker container we just built ([see: `docker tag`](https://docs.docker.com/engine/reference/commandline/tag/)). Below is an example of the output of the `s2d.save()` line we executed above.

```
Now attempting to run the command: 
[docker build --file /tmp/tmpywbu3_ad/Dockerfile 
 --tag classifier:iris /tmp/tmpywbu3_ad]
=====================================================================
> Sending build context to Docker daemon
> Step 1/6 : FROM python:3.6
> ---> c1e459c00dc3
... output truncated ...
> Step 6/6 : ENTRYPOINT python /code/api.py
> ---> Running in bd61983358d9
> Removing intermediate container bd61983358d9
> ---> fa2041ac6d60
> Successfully built fa2041ac6d60
> Successfully tagged classifier:iris
=====================================================================
Success! You can now run your Docker container using the following command:
	 docker run -d -p 5000:5000 classifier:iris
```

You can now test your container by asking it to predict the same Iris dataset and return the predicted probabilities ([see: `predict_proba`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.predict_proba)) as a DataFrame.

```python
from os import system
system("docker run -d -p 5000:5000 classifier:iris && sleep 5")

from requests import post
from pandas import read_json
request = post("http://localhost:5000/predict_proba/split", json=input_df.to_json(orient="split"))
result = read_json(request.content.decode(), orient="split")
print(result.head())
```

```
   setosa  versicolor  virginica
0       1         0.0        0.0
1       1         0.0        0.0
2       1         0.0        0.0
3       1         0.0        0.0
4       1         0.0        0.0
```

You can also request regular classification ([see: `predict`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.predict)). The format for the URL for your Docker container is as so:

```
http://[a]:[b]/[c]/[d]

a: the hostname of the container, defaults to `localhost`
b: the port of the container, defaults to 5000
c: one of `predict` or `predict_proba`, similar to the scikit-learn api
d: defaults to `split`; orient of the Pandas DataFrame JSON conversion*
```

(*: see [this documentation article](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_json.html) for more information about Pandas orients, and [this Github issue](https://github.com/pandas-dev/pandas/issues/18912#issuecomment-354430046) for a comparison; most of the time, setting the orient to `split` should do just fine)

```python
request = post(
    "http://localhost:5000/predict/split", 
    json=input_df.to_json(orient="split")
)
```

```
  prediction
0     setosa
1     setosa
2     setosa
3     setosa
4     setosa
```