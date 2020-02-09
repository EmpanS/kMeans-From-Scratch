# kMeans

### A numpy implementation of the k-means algorithm.
This is just-for-fun project where I implemented the k-means algorithm from scratch using numpy. Some time back, I decided to start implementing machine learning algorithms from scratch to get a better understanding of how they work.


## Content
This project contains three files:
1. Conda environment (.yml file): [kMeans.yml](https://github.com/EmpanS/kMeans-From-Scratch/blob/master/kMeans.yml)
2. The Class kMeans (.py file), contains the k-means algorithm: [kMeans.py](https://github.com/EmpanS/ML-from-Scratch/blob/master/kMeans.py)
3. An example notebook, containing two toy examples on how to use the class: [example.ipynb](https://github.com/EmpanS/ML-from-Scratch/blob/master/example.ipynb)

## How to use
1. Clone the repository.
2. Create the conda enviroment from the environment file called kMeans.yml by running:
```console
$ conda env create -f kMeans.yml
```
3. Then activate the enviornment:
```console
$ conda activate kMeans
```
4. Now, you can either go through the example iPython notebook or play with the kMeans class by simply importing the class. Example:
```python
from kMeans import kMeans

model = kMeans(k=3, X=data)
wss, predictions = model.fit()
```

## Further Improvements
These are my ideas for further improvements and possible extensions:
- Better error handling, there current error handling functionality is limited.
- Extended features, more methods to extract data from an instance of a class.
- Create a method for hyper-parameter optimization. Could include creation of a Scree-plot and Silhouette-functionality. 

## Lessons Learned
In this project I learned about the convenience of using virtual environments like conda. Instead of having to explicitly state what version of each library to use, one can just share the environment, that is, share the .yml file. Further, the biggest problem I encountered was about how to initiate the algorithm. I've learned that one should randomly assign each observation a label, but with that implementation I found that the algorithm quite often did not converge on the Iris-dataset when k>6. To solve this problem, I tried to instead start by setting the cluster centers as observations. It was a significantly improvement and I believe this methodology is best practice.

Any comments, suggestions or feedback is heavily appreciated. Thanks and happy clustering!

Emil Sandström
