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

### Functionalities

KValid uses Silhouette-Index and Elbow to validate the SimpleKMeans algorithm. Besides calculating
the SI and SSE, the package tell which is the best K and allows plot the graph into the screen and
save as PNG format.

### How to install

In order to install KValid, download trough the release menu in GitHub, [this](https://github.com/Theldus/KValid/releases/download/1.0.0/KValid.zip) link to be more specific.
(make sure that your WEKA version is >= 3.7.1)

Once downloaded, install through the package manager:
*Tools -> Package manager -> File/URL button -> Browse* (searches for the KValid.zip file) and click OK.

A confirmation message like 'Weka will need to be restarted after installation...", will appears. Just close and open Weka
again and the package should works.

Since KValid is a cluster evaluator, you can find it in the Cluster menu.
