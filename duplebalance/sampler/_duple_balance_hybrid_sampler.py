"""Class to perform duple-balanced hybrid-sampling."""

# Authors: Anon.
# License: MIT

# %%


import numbers
import numpy as np

from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing


from .base import BaseSampler
from ..utils._validation_param import check_pred_proba, check_type
from ..utils._validation import _deprecate_positional_args, check_target_type


# # For local test
# import sys
# sys.path.append("../..")
# from sampler.base import BaseSampler
# from utils._validation_param import check_pred_proba, check_type
# from utils._validation import _deprecate_positional_args, check_target_type



class DupleBalanceHybridSampler(BaseSampler):

    _sampling_type = "hybrid-sampling"

    @_deprecate_positional_args
    def __init__(
        self, *, 
        how='shem',
        sampling_strategy="auto", 
        k_bins=5,
        replacement=True, 
        random_state=None, 
    ):
        super().__init__(sampling_strategy=sampling_strategy)

        self.k_bins = k_bins
        self.replacement = replacement
        self.random_state = random_state
        self.how = how


    def _check_X_y(self, X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = self._validate_data(
            X,
            y,
            reset=True,
            accept_sparse=["csr", "csc"],
            dtype=None,
            force_all_finite=False,
        )
        return X, y, binarize_y


    @_deprecate_positional_args
    def fit_resample(self, X, y, *, sample_weight, **kwargs):
        return super().fit_resample(X, y, sample_weight=sample_weight, **kwargs)


    @_deprecate_positional_args
    def _fit_resample(self, X, y, *, 
                      y_pred_proba, 
                      i_estimator:int, 
                      total_estimator:int,
                      classes_, 
                      sample_weight=None,):

        # Check parameters
        k_bins_ = check_type(self.k_bins, 'k_bins', numbers.Integral)
        if k_bins_ <= 0:
            raise ValueError(
                f"'k_bins' must be an integer and > 0, got {k_bins_}."
            )
        self.k_bins_ = k_bins_
        self.replacement_ = check_type(self.replacement, 'replacement', bool)
        self.how_ = check_type(self.how, 'how', str)
        
        n_samples, n_classes = X.shape[0], classes_.shape[0]
        how = self.how_

        # Check random_state and predict probabilities
        random_state = check_random_state(self.random_state)
        y_pred_proba = check_pred_proba(y_pred_proba, n_samples, n_classes, dtype=np.float64)

        indexes = np.arange(n_samples)
        index_list = []
        
        self._weights = np.ones_like(y, dtype=float)

        # For each class C
        for i_c, target_class in zip(range(len(classes_)), classes_):
            if target_class in self.sampling_strategy_.keys():

                # Get the desired & actual number of samples of class C
                # and the index mask of class C
                n_target_samples_c = self.sampling_strategy_[target_class]
                class_index_mask = y == target_class
                n_samples_c = np.count_nonzero(class_index_mask)

                # Compute the error array
                error_c=np.abs(
                    np.ones(n_samples_c) - y_pred_proba[class_index_mask, i_c])

                # index_c: absolute indexes of class C samples
                index_c = indexes[class_index_mask]
                
                # Get the absolute indexes of resampled class C samples
                index_c_result = self._sample_single_class(
                    how=how,
                    error_c=error_c,
                    n_target_samples_c=n_target_samples_c,
                    index_c=index_c,
                    i_estimator=i_estimator,
                    total_estimator=total_estimator,
                    random_state=random_state)

                index_list.append(index_c_result)
        
        # Concatenate the result
        index_resampled = np.hstack(index_list)

        # Store the final undersample indexes
        self.sample_indices_ = index_resampled
        
        X_resampled, y_resampled = X[index_resampled], y[index_resampled]

        # Return the resampled X, y
        # also return resampled sample_weight (if sample_weight is not None)
        if sample_weight is not None:
            # sample_weight is already validated in super().fit_resample()
            weights_under = _safe_indexing(sample_weight, index_resampled)
            return X_resampled, y_resampled, weights_under
        else: return X_resampled, y_resampled

    def _sample_single_class(self, how:str, **kwargs):
        if how == 'shem':
            return self._hybrid_sample_single_class(**kwargs)
        elif how == 'hem':
            return self._error_sample_single_class(**kwargs)
        elif how == 'uniform':
            return self._uniform_sample_single_class(**kwargs)
        else: raise ValueError(f'Weighting strategy {how} is not supported.')
    
    def _error_sample_single_class(self, error_c, **kwargs):
        return self._weighted_sample_single_class(error_c, **kwargs)
    
    def _uniform_sample_single_class(self, error_c, **kwargs):
        return self._weighted_sample_single_class(np.ones_like(error_c), **kwargs)
        
    def _weighted_sample_single_class(self, weights, n_target_samples_c, 
                                   index_c, i_estimator, total_estimator, 
                                   random_state, replacement=None):
        replacement = self.replacement_ if replacement is None else replacement
        sample_proba = weights / weights.sum()
        return random_state.choice(
            index_c,
            size=n_target_samples_c,
            replace=replacement,
            p=sample_proba,)

    def _hybrid_sample_single_class(self, error_c, n_target_samples_c, 
                                    index_c, i_estimator, total_estimator, 
                                    random_state, replacement=None):
        """Perform duple-balanced resampling in a single class"""
        k_bins = self.k_bins_
        n_samples_c = error_c.shape[0]
        replacement = self.replacement_ if replacement is None else replacement
        replacement = True if n_samples_c < n_target_samples_c else replacement

        # if error is not distinguishable or no sample will be dropped
        if error_c.max() == error_c.min():
            # perform random under-sampling
            return random_state.choice(
                index_c,
                size=n_target_samples_c,
                replace=replacement)

        with np.errstate(divide='ignore', invalid='ignore'):
            # compute population & error contribution of each bin
#             populations, edges = np.histogram(error_c, bins=k_bins)
            populations, edges = np.histogram(error_c, bins=k_bins, range=(0, 1))
            contributions = np.zeros(k_bins)
            index_bins = []
            for i_bin in range(k_bins):
                index_bin = ((error_c >= edges[i_bin]) & (error_c < edges[i_bin+1]))
                if i_bin == (k_bins-1):
                    index_bin = index_bin | (error_c==edges[i_bin+1])
                index_bins.append(index_bin)
                if populations[i_bin] > 0:
                    contributions[i_bin] = error_c[index_bin].mean()

            # compute the expected number of samples to be sampled from each bin
            alpha = np.tan(np.pi*0.5*(i_estimator/(max(total_estimator-1, 1))))
            bin_weights = 1. / (contributions + alpha)
            bin_weights[np.isnan(bin_weights)|np.isinf(bin_weights)] = 0
            n_target_samples_bins = n_target_samples_c * bin_weights / bin_weights.sum()
            n_invalid_samples = sum(n_target_samples_bins[populations==0])
            if n_invalid_samples > 0:
                n_valid_samples = n_target_samples_c-n_invalid_samples
                n_target_samples_bins *= n_target_samples_c / n_valid_samples
                n_target_samples_bins[populations==0] = 0
            n_target_samples_bins = n_target_samples_bins.astype(int)+1

        with np.errstate(divide='ignore', invalid='ignore'):
            # perform soft (weighted) self-paced under-sampling
            soft_db_bin_weights = n_target_samples_bins / populations
            soft_db_bin_weights[~np.isfinite(soft_db_bin_weights)] = 0

        soft_db_bin_weights /= soft_db_bin_weights[soft_db_bin_weights!=0].min()
        soft_db_bin_weights = np.sqrt(soft_db_bin_weights)
        # compute sampling probabilities
        soft_db_sample_proba = np.zeros_like(error_c)
        for i_bin in range(k_bins):
            soft_db_sample_proba[index_bins[i_bin]] = soft_db_bin_weights[i_bin]

        self._weights[index_c] = soft_db_sample_proba
        soft_db_sample_proba /= soft_db_sample_proba.sum()

        # sample with respect to the sampling probabilities
        return random_state.choice(
            index_c,
            size=n_target_samples_c,
            replace=replacement,
            p=soft_db_sample_proba,)
        

    def _more_tags(self):
        return {
            "X_types": ["2darray", "string", "sparse", "dataframe"],
            "sample_indices": True,
            "allow_nan": True,
        }