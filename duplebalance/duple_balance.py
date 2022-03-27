"""
DupleBalanceClassifier: An ensemble classifier that performs 
inter-class and intra-class balancing for class-imbalanced learning.
"""

# Authors: Anon.
# License: MIT

# %%


from collections import Counter
import numpy as np
import numbers
from math import ceil
from tqdm import tqdm

from .base import BaseImbalancedEnsemble, MAX_INT
from .sampler._duple_balance_hybrid_sampler import DupleBalanceHybridSampler
from .utils._validation_data import check_eval_datasets
from .utils._validation_param import check_train_verbose 
from .utils._validation_param import check_eval_metrics
from .utils._validation_param import check_type
from .utils._validation import _deprecate_positional_args

from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_random_state
from sklearn.utils.validation import has_fit_parameter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# # For local test
# import sys
# sys.path.append(".")
# from base import BaseImbalancedEnsemble, MAX_INT
# from sampler._duple_balance_hybrid_sampler import DupleBalanceHybridSampler
# from utils._validation_data import check_eval_datasets
# from utils._validation_param import check_train_verbose 
# from utils._validation_param import check_eval_metrics
# from utils._validation_param import check_type
# from utils._validation import _deprecate_positional_args


BALANCING_SCHEDULE_PARAMS_TYPE = {
    'origin_distr': dict,
    'target_distr': dict,
    'i_estimator': numbers.Integral,
    'total_estimator': numbers.Integral,
}


# Properties
_method_name = 'DupleBalanceClassifier'
_sampler_class = DupleBalanceHybridSampler

_solution_type = 'resampling'
_sampling_type = 'hybrid-sampling'
_ensemble_type = 'general'
_training_type = 'iterative'

_properties = {
    'solution_type': _solution_type,
    'sampling_type': _sampling_type,
    'ensemble_type': _ensemble_type,
    'training_type': _training_type,
}


class DupleBalanceClassifier(BaseImbalancedEnsemble):
    """Duple-balanced Ensemble (DuBE) Classifier for class-imbalanced learning.
    
    DuBE is an ensemble learning framework for multi-class imbalanced 
    classification. It simultaneously performs inter-class and intra-class
    balancing during the ensemble training. It is an easy-to-use solution for 
    class-imbalanced problems, features outstanding computing efficiency, 
    good performance, and wide compatibility with different learning models.

    Parameters
    ----------
    base_estimator : estimator object, default=None
        The base estimator to fit on self-paced under-sampled subsets 
        of the dataset. Support for sample weighting is NOT required, 
        but need proper ``classes_`` and ``n_classes_`` attributes. 
        If ``None``, then the base estimator is ``DecisionTreeClassifier()``.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    perturb_alpha : float or str, default="auto"
        The multiplier of the calibrated Gaussian noise that was add on the 
        sampled data. It determines the intensity of the perturbation-based 
        augmentation. If `'auto'`, perturb_alpha will be automatically tuned 
        using a subset of the given training data.

    k_bins : int, default=5
        The number of error bins that were used to approximate 
        error distribution. It is recommended to set it to 5. 
        One can try a larger value when the smallest class in the 
        data set has a sufficient number (say, > 1000) of samples.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    n_jobs : int, default=None
        The number of jobs to run in parallel for :meth:`predict`. 
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` 
        context. ``-1`` means using all processors. See `Glossary 
        <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`_
        for more details.

    random_state : int, RandomState instance or None, default=None
        Control the randomization of the algorithm.
        If the base estimator accepts a `random_state` attribute, a different
        seed is generated for each instance in the ensemble.
        Pass an ``int`` for reproducible output across multiple function calls.
        
        - If ``int``, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.

    verbose : int, default=0
        Controls the verbosity when predicting.

    Attributes
    ----------
    base_estimator : estimator
        The base estimator from which the ensemble is grown.

    base_sampler_ : DupleBalanceHybridSampler
        The base sampler.

    estimators_ : list of estimator
        The collection of fitted base estimators.

    samplers_ : list of DupleBalanceHybridSampler
        The collection of fitted samplers.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.
    
    estimators_n_training_samples_ : list of ints
        The number of training samples for each fitted 
        base estimators.

    Notes
    -----
    See :ref:`sphx_glr_auto_examples_usage_plot_basic_example.py` for an example.

    Examples
    --------
    >>> from duplebalance import DupleBalanceClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, n_classes=3,
    ...                            n_informative=4, weights=[0.2, 0.3, 0.5],
    ...                            random_state=0)
    >>> clf = DupleBalanceClassifier(random_state=0)
    >>> clf.fit(X, y)  # doctest: +ELLIPSIS
    DupleBalanceClassifier(...)
    >>> clf.predict(X)  # doctest: +ELLIPSIS
    array([...])
    """
    
    def __init__(self, 
        base_estimator=None, 
        n_estimators:int=50, 
        perturb_alpha:float=0.,
        k_bins:int=5, 
        estimator_params=tuple(),
        n_jobs=None,
        random_state=None,
        verbose=0,):
    
        super(DupleBalanceClassifier, self).__init__( 
            base_estimator=base_estimator, 
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            random_state=random_state,
            n_jobs=n_jobs, 
            verbose=verbose)

        self.__name__ = _method_name
        self.base_sampler = _sampler_class()
        self._sampling_type = _sampling_type
        self._sampler_class = _sampler_class
        self._properties = _properties

        self.k_bins = k_bins
        self.perturb_alpha = perturb_alpha


    @staticmethod
    def _compute_distribution_statistic(X, y):
        """Compute the data statistics for perturbation calibration."""
        data_stat = {}
        for label in np.unique(y):
            idx = y == label
            X_i = X[idx]
            mean = np.mean(X_i, axis=0)
            std = np.std(X_i, axis=0)
            cov = np.cov(X_i.T)
            data_stat[label] = {'mean': mean, 'std': std, 'cov': cov, 'count': idx.sum()}
        return data_stat
    
    def _pertub_data_augment(self, X, y, perturb_args, random_state):
        """Private function for perturbation-based data augmentation."""
        if perturb_args is None:
            return X, y
        ptb_alpha = perturb_args['alpha']
        ptb_stats = perturb_args['stats']
        ptb_type = perturb_args['type']
        
        X_pertub = np.zeros_like(X)
        for label in np.unique(y):
            stat = ptb_stats[label]
            idx = (y==label)
            if ptb_type == 'gaussian':
                X_pertub_label = random_state.multivariate_normal(
                    mean=np.zeros(X.shape[1]), 
                    cov=stat['cov'], 
                    size=idx.sum())
            elif ptb_type == 'uniform':
                X_pertub_label = random_state.uniform(
                    low=-stat['std'], 
                    high=stat['std'], 
                    size=(idx.sum(), X.shape[1]))
            else: raise ValueError(f'Unsupport ptb_how {ptb_how}')
            
            X_pertub_label *= ptb_alpha
            X_pertub[idx] = X_pertub_label
            
        return X + X_pertub, y
    
    @staticmethod
    def _get_resampling_target(origin_distr, how='raw'):
        """Private function for determining the number of samples from each class."""
        keys, values = origin_distr.keys(), origin_distr.values()
        n_samples = sum(values)
        n_min, n_max, n_ave = min(values), max(values), int(n_samples/len(keys))
        if how == 'raw':
            return origin_distr
        elif how == 'hybrid':
            return dict([(c, int(n_samples/len(keys))) for c in keys])
        elif how == 'under':
            return dict([(c, min(values)) for c in keys])
        elif how == 'over':
            return dict([(c, max(values)) for c in keys])
        elif how == 'inv':
            return {c: min(n_max - (origin_distr[c]-n_min), 3*n_min) for c in keys}
        else: raise RuntimeError(f'resampling_target: {how} is not supported.')
    
    @_deprecate_positional_args
    def fit(self, X, y, *, sample_weight=None, **kwargs):
        """Build a DuBE classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        resampling_target : {'hybrid', 'under', 'over', 'raw'}, default="hybrid"
            Determine the number of instances to be sampled from each class 
            (inter-class balancing).

            - If ``under``, perform under-sampling. The class containing the 
              fewest samples is considered the minority class :math:`c_{min}`.
              All other classes are then under-sampled until they are of the same
              size as :math:`c_{min}`.

            - If ``over``, perform over-sampling. The class containing the 
              argest number of  samples is considered the majority class 
              :math:`c_{maj}`. All other classes are then over-sampled until 
              they are of the same size as :math:`c_{maj}`.
            
            - If ``hybrid``, perform hybrid-sampling. All classes are 
              under/over-sampled to the average number of instances from each class.
            
            - If ``raw``, keep the original size of all classes when resampling.

        resampling_strategy : {'hem', 'shem', 'uniform'}, default="shem"
            Decide how to assign resampling probabilities to instances during 
            ensemble training (intra-class balancing).

            - If ``hem``, perform hard-example mining. Assign probability 
              with respect to instance's latest prediction error.

            - If ``shem``, perform soft hard-example mining. Assign probability 
              by inversing the classification error density.

            - If ``uniform``, assign uniform probability, i.e., random resampling.

        replacement : bool, default=True
            Whether samples are drawn with replacement. If False, sampling
            without replacement is performed (not applicable to over/hybrid-sampling).

        perturb_alpha : float or str, default="auto"
            The temporary value for perturb_alpha in a single function call.
            The multiplier of the calibrated Gaussian noise that was add on the 
            sampled data. It determines the intensity of the perturbation-based 
            augmentation. If `'auto'`, perturb_alpha will be automatically tuned 
            using a subset of the given training data.

        k_bins : int, default=5
            The temporary value for k_bins in a single function call.
            The number of error bins that were used to approximate 
            error distribution. It is recommended to set it to 5. 
            One can try a larger value when the smallest class in the 
            data set has a sufficient number (say, > 1000) of samples.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights for base classifier training. If None, 
            the sample weights are initialized to ``1 / n_samples``.
        
        eval_datasets : dict, default=None
            Dataset(s) used for evaluation during the ensemble training process.
            The keys should be strings corresponding to evaluation datasets' names. 
            The values should be tuples corresponding to the input samples and target
            values. 
            
            Example: ``eval_datasets = {'valid' : (X_valid, y_valid)}``
        
        eval_metrics : dict, default=None
            Metric(s) used for evaluation during the ensemble training process.

            - If ``None``, use the weighted (class-balanced) roc_auc_score by default.

            - If ``dict``, the keys should be strings corresponding to evaluation 
              metrics' names. The values should be tuples corresponding to the metric 
              function (``callable``) and additional kwargs (``dict``).

                - The metric function should at least take 2 positional arguments 
                  ``y_true``, ``y_pred``, and returns a ``float`` as its score. 
                - The metric additional kwargs should specify the additional arguments
                  that need to be passed into the metric function. 
            
            Example: 
            ``{'weighted_f1': (sklearn.metrics.f1_score, {'average': 'weighted'})}``
        
        train_verbose : bool, int or dict, default=True
            Controls the verbosity during ensemble training/fitting.

            - If ``bool``: ``False`` means disable training verbose. ``True`` means 
              print training information to sys.stdout use default setting:
              
                - ``'granularity'``        : ``int(n_estimators/10)``
                - ``'print_distribution'`` : ``True``
                - ``'print_metrics'``      : ``True``

            - If ``int``, print information per ``train_verbose`` rounds.

            - If ``dict``, control the detailed training verbose settings. They are:

                - ``'granularity'``: corresponding value should be ``int``, the training
                  information will be printed per ``granularity`` rounds.
                - ``'print_distribution'``: corresponding value should be ``bool``, 
                  whether to print the data class distribution 
                  after resampling. Will be ignored if the 
                  ensemble training does not perform resampling.
                - ``'print_metrics'``: corresponding value should be ``bool``, 
                  whether to print the latest performance score.
                  The performance will be evaluated on the training 
                  data and all given evaluation datasets with the 
                  specified metrics.
              
            .. warning::
                Setting a small ``'granularity'`` value with ``'print_metrics'`` enabled 
                can be costly when the training/evaluation data is large or the metric 
                scores are hard to compute. Normally, one can set ``'granularity'`` to 
                ``n_estimators/10``.

        Returns
        -------
        self : object
        """

        # Check random state
        self.random_state = check_random_state(self.random_state)

        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(X, y, **self.check_x_y_args)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
            sample_weight /= sample_weight.sum()
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight cannot contain negative weights")

        # Remap output
        n_samples, self.n_features_ = X.shape
        self.features_ = np.arange(self.n_features_)
        self._n_samples = n_samples
        self._validate_y(y)
        self._binary_flag = True if len(self.classes_) <= 2 else False

        # Check parameters
        self._validate_estimator(default=DecisionTreeClassifier())
        
        # If the base estimator do not support sample weight and sample weight
        # is not None, raise an ValueError
        support_sample_weight = has_fit_parameter(self.base_estimator_,
                                                "sample_weight")
        if not support_sample_weight and sample_weight is not None:
            raise ValueError("The base estimator doesn't support sample weight")

        self.estimators_, self.estimators_features_ = [], []

        return self._fit(X, y, sample_weight=sample_weight, **kwargs)

    @_deprecate_positional_args
    def _fit(self, X, y, 
            *,
            resampling_target='hybrid',
            resampling_strategy='shem',
            replacement=True,
            perturb_alpha=.0,
            k_bins=None,
            sample_weight=None, 
            eval_datasets:dict=None,
            eval_metrics:dict=None,
            train_verbose:bool or int or dict=False,
            ):
        
        self.perturb_alpha = perturb_alpha
        
        # X, y, sample_weight, base_estimators_ (default=DecisionTreeClassifier),
        # n_estimators, random_state, sample_weight are already validated in super.fit()
        random_state, n_estimators, classes_ = \
            self.random_state, self.n_estimators, self.classes_
        n_samples = y.shape[0]
        
        # if not using alternative k_bins/perturb_alpha
        if k_bins is None:
            k_bins = self.k_bins
            
        # Compute the data statistics for perturbation calibration
        data_stat = self._compute_distribution_statistic(X, y)
        self.data_stat_ = data_stat

        # Check the perturbation multiplier alpha
        perturb_alpha = self.perturb_alpha if perturb_alpha is None else perturb_alpha
        check_type(perturb_alpha, 'perturb_alpha', (numbers.Real, str))
        if isinstance(perturb_alpha, numbers.Real):
            if perturb_alpha < 0:
                raise ValueError(
                    f"When 'perturb_alpha' is a real number, it must > 0, got {perturb_alpha}."
                )
        elif perturb_alpha == 'auto':
            perturb_alpha = self._auto_tune_alpha(X.copy(), y.copy())
        else: raise TypeError(
            f"When 'perturb_alpha' is a string, it must be 'auto', got '{perturb_alpha}' instead."
        )
        self.perturb_alpha_ = perturb_alpha

        # Check evaluation data
        check_x_y_args = self.check_x_y_args
        self.eval_datasets_ = check_eval_datasets(eval_datasets, X, y, **check_x_y_args)

        # Set target sample strategy
        origin_distr_ = dict(Counter(y))
        target_distr_ = self._get_resampling_target(origin_distr_, how=resampling_target)
        self.origin_distr_, self.target_distr_ = origin_distr_, target_distr_

        # Check evaluation metrics
        self.eval_metrics_ = check_eval_metrics(eval_metrics)
        
        # Check training train_verbose format
        self.train_verbose_ = check_train_verbose(
            train_verbose, self.n_estimators, **self._properties)
        
        # Set training verbose format
        self._init_training_log_format()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimators_features_ = []
        self.estimators_n_training_samples_ = np.zeros(n_estimators, dtype=np.int)
        self.samplers_ = []
        self.y_pred_proba_latest = np.zeros((self._n_samples, self.n_classes_), 
                                             dtype=np.float64)
        
        # Genrate random seeds array
        self._seeds = random_state.randint(MAX_INT, size=n_estimators)
        seeds = [np.random.RandomState(seed) for seed in self._seeds]

        # Check if sample_weight is specified
        specified_sample_weight = (sample_weight is not None)

        # Perturbation setting
        perturb_args_ = {
            'alpha': perturb_alpha,
            'stats': data_stat,
            'type': 'gaussian', 
        }

        for i_iter in range(n_estimators):
            
            # Initialize a DupleBalanceHybridSampler
            sampler = self._make_sampler(
                append=True,
                how=resampling_strategy,
                sampling_strategy=target_distr_,
                k_bins=k_bins,
                replacement=replacement,
                random_state=seeds[i_iter],
            )
            
            # update self.y_pred_proba_latest
            self._update_cached_prediction_probabilities(i_iter, X)

            # Perform duple-balanced hybrid-sampling
            resample_out = sampler.fit_resample(X, y, 
                    y_pred_proba=self.y_pred_proba_latest,
                    classes_=classes_,
                    i_estimator=i_iter,
                    total_estimator=n_estimators,
                    sample_weight=sample_weight,
                    )
                    
            if specified_sample_weight:
                (X_resampled, y_resampled, sample_weight_resampled) = resample_out
            else: (X_resampled, y_resampled) = resample_out
            
            
            # Perturbation-based data augmentation
            X_augmented, y_augmented = self._pertub_data_augment(
                X_resampled, y_resampled, perturb_args_, random_state=seeds[i_iter])

            # Train a new base estimator on resampled and augmented data 
            # and add it into self.estimators_
            estimator = self._make_estimator(append=True, random_state=seeds[i_iter])
            if specified_sample_weight:
                estimator.fit(X_augmented, y_augmented, 
                              sample_weight=sample_weight_resampled)
            else: estimator.fit(X_augmented, y_augmented)

            # Record training info
            self.estimators_features_.append(self.features_)
            self.estimators_n_training_samples_[i_iter] = y_resampled.shape[0]

            # Print training log to stdout
            self._training_log_to_console(i_iter, y_resampled)
        
        return self
    

    def _update_cached_prediction_probabilities(self, i_iter, X):
        """Private function that maintains a latest prediction probabilities of the training
         data during ensemble training. Must be called in each iteration before fit the 
         base_estimator."""
        if i_iter == 0:
            self.y_pred_proba_latest = np.zeros((X.shape[0], self.n_classes_), 
                                                dtype=np.float64)
        else:
            y_pred_proba_latest = self.y_pred_proba_latest
            y_pred_proba_new = self.estimators_[-1].predict_proba(X)
            self.y_pred_proba_latest = (y_pred_proba_latest * i_iter + y_pred_proba_new) / (i_iter+1)
        return
    

    def _stratified_random_sampling(self, X, y, n_samples_resample):
        """Class-wise random sampling that guarantees same class distribution."""
        n_samples_all = y.shape[0]
        resample_distr = dict(
            (label, int(n_samples_c * n_samples_resample / n_samples_all))
            for label, n_samples_c in Counter(y).items()
        )
        index_all = np.arange(n_samples_all)
        index_resample = []
        for label in self.classes_:
            index_c = index_all[y==label]
            index_resample_c = self.random_state.choice(
                index_c,
                size=resample_distr[label],
                replace=False,
            )
            index_resample.append(index_resample_c)
        
        index_resample = np.hstack(index_resample)
        
        return X[index_resample], y[index_resample]
    
    
    def _auto_tune_alpha(self, X, y):
        """Automatically tune the perturbation parameter alpha using a small subset."""
        n_max_samples_valid  = 1000
        if y.shape[0] > n_max_samples_valid:
            X, y = self._stratified_random_sampling(
                X, y, n_max_samples_valid)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        search_space = np.linspace(0, 0.5, 21)
        search_scores, search_stds = [], []
        iterations = tqdm(search_space)
        iterations.set_description("'perturb_alpha' == 'auto', auto tuning")
        self.n_estimators, n_estimators_raw = 5, self.n_estimators
        for alpha in iterations:
            cv_scores = []
            self.perturb_alpha = alpha
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                self._fit(X_train, y_train)
                cv_scores.append(self.score(X_test, y_test))
            search_scores.append(np.mean(cv_scores))
            search_stds.append(np.std(cv_scores))

        sorted_results = sorted(zip(search_space, search_scores, search_stds), key=lambda k: k[1])
        best_alpha, best_score, _ = sorted_results[-1]
        self._auto_tune_results = sorted_results
        print ("\nThe perturb_alpha will be set to {:.3f} with {:.3f} balanced AUROC (validation score)".format(
            best_alpha, best_score
            ))

        # Set the alpha to best_alpha
        self.perturb_alpha = best_alpha
        # Recover original n_estimators
        self.n_estimators = n_estimators_raw

        return best_alpha
    

    def score(self, X, y):
        """
        Return the balanced AUROC score on the given test data and labels.
        In multi-label imbalanced classification, this is an unbiased metric.
        The roc_auc score is calculated for each label, and find their average, 
        weighted by support (the number of true instances for each label).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            Balanced AUROC score of ``self.predict_proba(X)`` wrt. `y`.
        """
        if self._binary_flag:
            kwargs = {'y_score': self.predict_proba(X)[:, 1]}
        else:
            kwargs = {
                'y_score': self.predict_proba(X),
                'multi_class': 'ovo',
            }

        return roc_auc_score(
            y, average = 'macro', **kwargs
        )

# %%

if __name__ == '__main__':
    from collections import Counter
    from copy import copy
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    
    # X, y = make_classification(n_classes=2, class_sep=2, # 2-class
    #     weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    #     n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    X, y = make_classification(n_classes=3, class_sep=2, # 3-class
        weights=[0.1, 0.3, 0.6], n_informative=3, n_redundant=1, flip_y=0,
        n_features=20, n_clusters_per_class=1, n_samples=2000, random_state=10)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=42)

    origin_distr = dict(Counter(y_train)) # {2: 600, 1: 300, 0: 100}
    print('Original training dataset shape %s' % origin_distr)

    target_distr = {2: 200, 1: 100, 0: 100}

    init_kwargs_default = {
        'base_estimator': None,
        'n_estimators': 100,
        'k_bins': 5,
        'estimator_params': tuple(),
        'n_jobs': None,
        'random_state': 42,
        'verbose': 0,
    }

    fit_kwargs_default = {
        'X': X_train,
        'y': y_train,
        'sample_weight': None,
        'eval_datasets': {'valid': (X_valid, y_valid)},
        'eval_metrics': {
            'acc': (accuracy_score, {}),
            'balanced_acc': (balanced_accuracy_score, {}),
            'weighted_f1': (f1_score, {'average':'weighted'}),},
        'train_verbose': {
            'granularity': 10,
            'print_distribution': True,
            'print_metrics': True,},
    }

    ensembles = {}

    init_kwargs, fit_kwargs = copy(init_kwargs_default), copy(fit_kwargs_default)
    clf = DupleBalanceClassifier(**init_kwargs).fit(**fit_kwargs)

    # %%
