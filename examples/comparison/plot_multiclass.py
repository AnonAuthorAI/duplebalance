"""
==============================================================
Testing DuBE with different number of classes (3-15)
==============================================================

In this example, we compare the :class:`duplebalance.DupleBalanceClassifier` 
and other ensemble-based class-imbalanced learning methods on multi-class
tasks (with number of classes varying from 3 to 15).
"""

# %%
print(__doc__)

RANDOM_STATE = 42

# %% [markdown]
# Preparation
# -----------
# Import necessary packages.

from duplebalance import DupleBalanceClassifier
from duplebalance.base import sort_dict_by_key

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# %% [markdown]
# Train All Ensemble Classifier
# ----------------------------------------------------------
# Train all ensemble-based IL classifier (including DuBE) on multi-class datasets.

from imbalanced_ensemble.ensemble import *

ensemble_init_kwargs = {
    'base_estimator': DecisionTreeClassifier(),
    'n_estimators': 10,
    'random_state': RANDOM_STATE,
}
dube_fit_kwargs = {
    'resampling_target': 'hybrid',
    'resampling_strategy': 'shem',
    'perturb_alpha': .5,
}
eval_kwargs = {'average': 'macro', 'multi_class': 'ovo'}

ensemble_clfs = {
    'DuBE': DupleBalanceClassifier,
    'RusBoost': RUSBoostClassifier,
    'OverBoost': OverBoostClassifier,
    'SmoteBoost': SMOTEBoostClassifier,
    'RusBoost': RUSBoostClassifier,
    'UnderBagging': UnderBaggingClassifier,
    'OverBagging': OverBaggingClassifier,
    'SmoteBagging': SMOTEBaggingClassifier,
    'Cascade': BalanceCascadeClassifier,
    'SelfPacedEns': SelfPacedEnsembleClassifier,
}

# Initialize results list
all_results = []

for n_class in range(3, 16):
    
    # Assign long-tail class weights
    weights = np.array([np.power(.8, i) for i in range(n_class)])
    weights /= weights.sum()
    info = "#Classes: {}\nImbalance Ratio: ".format(n_class)
    for weight in weights:
        info += '{:.2f}/'.format(weight/weights.min())
    print (info.rstrip('/'))
    
    # Generate synthetic multi-class imbalanced dataset
    X, y = make_classification(n_classes=n_class, class_sep=1,
        weights=weights, n_informative=4, n_redundant=1, flip_y=0,
        n_features=20, n_clusters_per_class=1, n_samples=5000, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    for ens_name, clf_class in ensemble_clfs.items():
        
        # Train all ensemble classifiers
        clf = clf_class(
            **ensemble_init_kwargs
        )
        if ens_name == 'DuBE':
            clf.fit(X_train, y_train, **dube_fit_kwargs)
        else: clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)
        score = roc_auc_score(y_test, y_pred_proba, **eval_kwargs)
        all_results.append([ens_name, score, n_class])
        print ("{:<15s} | Balanced AUROC: {:.3f}".format(ens_name, score))

# %% [markdown]
# Results Visualization
# --------------------------

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')

all_results_columns = ['Method', 'AUROC (macro)', '#Classes']
data_vis = pd.DataFrame(all_results, columns=all_results_columns)


def plot_results_comp(data_vis, x, y, title, figsize=(8,6)):
    fig = plt.figure(figsize=figsize)
    ax = sns.lineplot(
        data=data_vis, x=x, y=y, hue='Method', style='Method',
        markers=True, err_style='bars', linewidth=4, markersize=20, alpha=0.9
    )
    for position, spine in ax.spines.items():
        spine.set_color('black')
        spine.set_linewidth(2)
    ax.grid(color = 'black', linestyle='-.', alpha=0.3)
    ax.set_ylabel('AUROC (macro)')
    ax.set_title(title)
    ax.legend(
        title='',
        borderpad=0.25,
        columnspacing=0.05,
        borderaxespad=0.15,
        handletextpad=0.05,
        labelspacing=0.05,
        handlelength=1.2,
        )
    return ax

plot_results_comp(data_vis, x='#Classes', y='AUROC (macro)',
                  title='DuBE versus Ensemble Baselines (#Classes 3-15)')