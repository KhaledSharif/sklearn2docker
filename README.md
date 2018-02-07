# sklearn2docker
#### Convert your trained scikit-learn classifier to a Docker container with a pre-configured API

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](http://www.gnu.org/licenses/lgpl-3.0)

## Getting started

First, create your `sklearn` classifier. In this example we will use the iris dataset.

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
    class_names=iris['target_names'].tolist(),
)
s2d.save()
```

Below is an example of the output of the `s2d.save()` line we executed above.


```
Now attempting to run the command: 
[docker build --file /tmp/tmpywbu3_ad/Dockerfile 
 --tag classifier:1518019754 /tmp/tmpywbu3_ad]
=====================================================================
> Sending build context to Docker daemon  8.192kB
> Step 1/6 : FROM python:3.6
> ---> c1e459c00dc3
... output truncated ...
> Step 6/6 : ENTRYPOINT python /code/api.py
> ---> Running in bd61983358d9
> Removing intermediate container bd61983358d9
> ---> fa2041ac6d60
> Successfully built fa2041ac6d60
> Successfully tagged classifier:1518019754
=====================================================================
Success! You can now run your Docker container using the following command:
	 docker run -d -p 5000:5000 classifier:1518019754
```

