### KValid
A simple clustering evaluation of KMeans for the Weka

-----------------

<p align="center">
	<img align="center" src="https://i.imgur.com/4IK452f.png" alt="Silhouette-Index in IRIS dataset">
	<br>
	<i>Silhouette-Index in IRIS dataset</i>
</p>

### What is KValid?

KValid is a simple clustering evaluation package for [WEKA](http://www.cs.waikato.ac.nz/ml/weka/).
It uses the SimpleKMeans algorithm as a backend to cluster the instances and evaluates
the clusterer using some algorithms, currently Silhouette-Index and Elbow.

### Funcionalities

KValid uses Silhouette-Index and Elbow to validate the SimpleKMeans algorithm. Besides calculating
the SI and SSE, the package tell which is the best K and allows plot the graph into the screen and
save as PNG format.
