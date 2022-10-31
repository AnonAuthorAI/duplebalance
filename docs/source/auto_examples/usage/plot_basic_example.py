"""
==============================================================
Basic usage example of `DupleBalanceClassifier`
==============================================================

This example shows the basic usage of :class:`duplebalance.DupleBalanceClassifier`.
"""

# %%
print(__doc__)

RANDOM_STATE = 42

# %% [markdown]
# Preparation
# -----------
# First, we will import necessary packages and generate an example
# multi-class imbalanced dataset.

from duplebalance import DupleBalanceClassifier
from duplebalance.base import sort_dict_by_key
from duplebalance.utils._plot import plot_2Dprojection_and_cardinality

from collections import Counter
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# %% [markdown]
# Make a 5-class imbalanced classification task

X, y = make_classification(n_classes=5, class_sep=1, # 5-class
    weights=[0.05, 0.05, 0.15, 0.25, 0.5], n_informative=10, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1, n_samples=2000, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

origin_distr = sort_dict_by_key(Counter(y_train))
test_distr = sort_dict_by_key(Counter(y_test))
print('Original training dataset shape %s' % origin_distr)
print('Original test dataset shape %s' % test_distr)

# Visualize the dataset
projection = KernelPCA(n_components=2).fit(X, y)
fig = plot_2Dprojection_and_cardinality(X, y, projection=projection)
plt.show()

# %% [markdown]
# Train a DupleBalance Classifier
# --------------------------------------------------
# Basic usage of DupleBalanceClassifier

# Train a DupleBalanceClassifier
clf = DupleBalanceClassifier(
    n_estimators=5,
    random_state=RANDOM_STATE,
).fit(X_train, y_train)

# Predict & Evaluate
score = clf.score(X_test, y_test)
print ("DupleBalance {} | Balanced AUROC: {:.3f} | #Training Samples: {:d}".format(
    len(clf.estimators_), score, sum(clf.estimators_n_training_samples_)
    ))

# %% [markdown]
# Train DupleBalanceClassifier with automatic parameter tuning

# Train a DupleBalanceClassifier
clf = DupleBalanceClassifier(
    n_estimators=5,
    random_state=RANDOM_STATE,
).fit(
    X_train, y_train,
    perturb_alpha='auto',
)

# Predict & Evaluate
score = clf.score(X_test, y_test)
print ("DupleBalance {} | Balanced AUROC: {:.3f} | #Training Samples: {:d}".format(
    len(clf.estimators_), score, sum(clf.estimators_n_training_samples_)
    ))

# %% [markdown]
# Train DupleBalanceClassifier with advanced training log

# Train a DupleBalanceClassifier
clf = DupleBalanceClassifier(
    n_estimators=5,
    random_state=RANDOM_STATE,
).fit(
    X_train, y_train,
    perturb_alpha='auto',
    eval_datasets={'test': (X_test, y_test)},
    train_verbose={
        'granularity': 1,
        'print_distribution': True,
        'print_metrics': True,
    },
)

# Predict & Evaluate
score = clf.score(X_test, y_test)
print ("DupleBalance {} | Balanced AUROC: {:.3f} | #Training Samples: {:d}".format(
    len(clf.estimators_), score, sum(clf.estimators_n_training_samples_)
    ))