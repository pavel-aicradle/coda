# CODA
TODO: Continuous integration status button

TODO: Sphinx Documentation, [How it works](https://safexai.atlassian.net/wiki/spaces/PROG/pages/1582399552/Community+Outlier+Detection+Algorithm+Notes+and+Proofs) (private TODO: make public?)

This repository represents an implementation of the Community Outlier Detection Algorithm (CODA), invented by Gao and coauthors[1]. This is an algorithm for finding anomalous nodes *in static graphs*, which I encountered while reading a survey on graph-based anomaly detection[2].

The algorithm looks for nodes that are *unlike the communities they're connected to*. That is, it does *not* look for global outliers, local outliers, or outliers within connection-based clusters. Instead it attempts to find `K` clusters and `n` number or portion of anomalous nodes, where the outliers do not affect the characterization of the clusters they're connected to.

This implementation can handle node parameters of both numeric and categorical type.

## Installation and Usage

TODO: push wheel up to the cheese shop

```shell
python3 -m pip install <TODO>
```

Or clone down the repo and

```shell
python3 -m pip install /path/to/repo
```

Or clone down the repo and

```shell
python3 setup.py install
```

TODO: some usage pointers after I've played with it and gotten a feel


## Technical Explanation

This algorithm is based on modeling the problem as a Hidden Markov Random Field. "Hidden" means we assume each node has a random variable `z_i` that denotes its cluster, `k`, which we can't observe. "Markov" means we're assuming this obeys the Markov Property, whereby a node's cluster assignment is only influenced by its own attributes and the assignments of its immediate neighbors. And "random field" refers to the generator functions (one per cluster) that give rise to attribute data, `s_i`, for each node, which we model as a sample of observable random variable `x_i`.

The model is optimized using [Expectation Maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm), wherein clusters' generator distributions are parameterized by `Theta`, and nodes have assignments `Z`. Since `Theta` and `Z` depend on each other, the algorithm proceeds in alternating fashion toward a local minimum "energy". Energy for this problem depends on how well a node's attributes fit cluster distributions, and on whether its assignment matches that of its neighbors, weighted by strength of relationship to those neighbors.

Choice of generator distribution to model clusters is up to the user, but it affects which parameters need to be optimized and how to find their updated values. This library provides two choices: `'independent'` and `'distance'`.
- The `'independent'` generator distribution considers the joint distribution over all node attributes to be equal to the product of univariate distributions over each parameter. That is, `P(A=a,B=b,C=c) == P(A=a)P(B=b)P(C=c)`, and `P(A)`, `P(B)`, and `P(C)` all get their own parameters. These univariate distributions are Gaussians for numerical random variables and Multinomials for categorical variables. (See equations 3 and 4 in [1].)
- For the `'distance'` generator distribution `P(A=a,B=b,C=c) == e^-D((A,B,C) - mu)` (See "Extensions" on page 7 of [1].), where `D` is a distance function. Finding a distance function that works over a mix of both numeric and categorical attributes is nontrivial, but several options are laid out in [3]. This implementation uses the "Regular Simplex Method" to transform categorical variables into equidistant points in multidimensional space. These vectors are stacked together with the numeric variables to yield a purely numeric representation. The mean and covariance of these are used with a [Mahalanobis-like formula](https://en.wikipedia.org/wiki/Mahalanobis_distance#Definition_and_properties) to produce a distance.

## References

1. Jing Gao et al. (2010). "On Community Outliers and their Efficient
Detection in Information Networks" https://cse.buffalo.edu/~jing/doc/kdd10_coda.pdf
2. Leman Akoglu, Hanghang Tong, & Danai Koutra. (2014). "Graph-based Anomaly Detection and Description: A Survey" https://arxiv.org/pdf/1404.4679.pdf
3. Brendan McCane and Michael Albert. (2007). "Distance Functions for Categorical and Mixed Variables" https://www.cs.otago.ac.nz/staffpriv/mccane/publications/distance_categorical.pdf