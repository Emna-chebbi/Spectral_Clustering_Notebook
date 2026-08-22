# Spectral Clustering

A hands-on notebook exploring **Spectral Clustering** — what it is, how it works mathematically, and how it performs against a K-Means baseline on the classic scikit-learn Wine dataset.

## Overview

This project walks through the full unsupervised learning workflow:

- **Theory first**: an intuitive + mathematical explanation of Spectral Clustering (similarity graphs, the graph Laplacian, eigenvectors, and how K-Means is applied in the transformed space).
- **A visual proof of concept**: two interleaving crescents (`make_moons`) where Spectral Clustering perfectly separates the data (ARI = 1.000) while K-Means fails (ARI = 0.247).
- **A real dataset**: the Wine dataset (178 samples, 13 chemical features, 3 cultivars), clustered using only chemical measurements — the true labels are held out and used solely for evaluation.
- **A rigorous ML workflow**: proper train/validation/test splitting, hyperparameter tuning (`k` and `n_neighbors`) on the validation set only, and a final unbiased evaluation on a held-out test set.
- **Handling Spectral Clustering's transductive nature**: since it can't natively predict on new points, a K-Nearest Neighbors classifier is trained on (features → discovered cluster) to extend cluster assignments to unseen wines.

## Key Results

| Method | Test Silhouette | Test ARI | Test NMI |
|---|---|---|---|
| **Spectral Clustering** | 0.280 | **1.000** | **1.000** |
| K-Means (baseline) | 0.272 | 0.909 | 0.909 |

Using only chemical features (no labels during training), Spectral Clustering perfectly recovered the three real grape cultivars on unseen test wines, outperforming the K-Means baseline.

## Methodology Notes

- **Standardization**: All 13 chemical features are standardized (fit on training data only) before clustering, since features like `proline` have a much larger numeric range than features like `hue`, and would otherwise dominate the similarity computation.
- **No data leakage**: The true cultivar labels are never used to fit any clustering model — only to evaluate results after the fact.
- **Model selection discipline**: `k` and `n_neighbors` are chosen using validation-set ARI only. The test set is touched exactly once, at the very end.
- **Out-of-sample extension**: Because Spectral Clustering has no native `.predict()`, a K-NN classifier (`n_neighbors=5`) trained on (training features → training cluster labels) assigns clusters to validation/test wines.

## Final Chosen Parameters

- `k = 3` clusters
- `n_neighbors = 5` (for the nearest-neighbors similarity graph)
- `affinity = "nearest_neighbors"`
- `assign_labels = "kmeans"`

## Acknowledgments

- Dataset: [`sklearn.datasets.load_wine`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)
- Built with [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/), and [seaborn](https://seaborn.pydata.org/)
