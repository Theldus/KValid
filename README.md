### KValid
A simple clustering evaluation of KMeans for the Weka

-----------------

<p align="center">
	<img align="center" src="https://i.imgur.com/pawYbDU.png" alt="Silhouette-Index in IRIS dataset">
	<br>
	<i>Silhouette-Index in IRIS dataset</i>
</p>

### What is KValid?

KValid is a simple clustering evaluation package for [WEKA](http://www.cs.waikato.ac.nz/ml/weka/). It uses the
SimpleKMeans algorithm as a backend to cluster the instances and evaluates
the clusterer using some algorithms, currently only Silhouette-Index.

Note that the package does not tell you which is the best K but only output of the metric,
which can give you a clue to which is the best K. Also, keep in mind that these metrics
can sometimes induce an incorrect K, therefore, it is important to manually review your
dataset to get some better information.
