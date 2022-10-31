"""Utilities for parameter validation."""

# Authors: Anon.
# License: MIT


from copy import copy
from warnings import warn
from collections import Counter

import numbers
import numpy as np
from math import ceil
from sklearn.ensemble import BaseEnsemble
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import roc_auc_score


EVAL_METRICS_DEFAULT = {
    'balance-roc-auc': (roc_auc_score, {'average': 'weighted', 'multi_class': 'ovo'}),
}
EVAL_METRICS_INFO = \
            " Example 'eval_metrics': {..., 'metric_name': ('metric_func', 'metric_kwargs'), ...}."
            # " where `metric_name` is string, `metric_func` is `callable`," + \
            # " and `metric_arguments` is a dict of arguments" + \
            # " that needs to be passed to the metric function," + \
            # " e.g., {..., `argument_name`: `value`}."
EVAL_METRICS_TUPLE_TYPE = {
    'metric_func': callable,
    'metric_kwargs': dict,
}
EVAL_METRICS_TUPLE_LEN = len(EVAL_METRICS_TUPLE_TYPE)


def _check_eval_metric_func(metric_func):
    if not callable(metric_func):
        raise TypeError(
            f" The 'metric_func' should be `callable`, got {type(metric_func)},"
            f" please check your usage."
            + EVAL_METRICS_INFO
        )
    return metric_func


def _check_eval_metric_args(metric_kwargs):
    if not isinstance(metric_kwargs, dict):
        raise TypeError(
            f" The 'metric_kwargs' should be a `dict` of arguments"
            f" that needs to be passed to the metric function,"
            f" got {type(metric_kwargs)}, "
            f" please check your usage."
            + EVAL_METRICS_INFO
        )
    return metric_kwargs


def _check_eval_metric_name(metric_name):
    if not isinstance(metric_name, str):
        raise TypeError(
            f" The keys must be `string`, got {type(metric_name)}, "
            f" please check your usage."
            + EVAL_METRICS_INFO
        )
    return metric_name


def _check_eval_metric_tuple(metric_tuple, metric_name):
    if not isinstance(metric_tuple, tuple):
        raise TypeError(
            f" The value of '{metric_name}' is {type(metric_tuple)} (should be tuple)," + \
            f" please check your usage."
            + EVAL_METRICS_INFO
        )
    elif len(metric_tuple) != EVAL_METRICS_TUPLE_LEN:
        raise ValueError(
            f" The data tuple of '{metric_name}' has {len(metric_tuple)} element(s)" + \
            f" (should be {EVAL_METRICS_TUPLE_LEN}), please check your usage."
            + EVAL_METRICS_INFO
        )
    else:
        return (
            _check_eval_metric_func(metric_tuple[0]),
            _check_eval_metric_args(metric_tuple[1]),
        )


def _check_eval_metrics_dict(eval_metrics_dict):
    """check 'eval_metrics' dict."""
    eval_metrics_dict_ = {}
    for metric_name, metric_tuple in eval_metrics_dict.items():
        
        metric_name_ = _check_eval_metric_name(metric_name)
        metric_tuple_ = _check_eval_metric_tuple(metric_tuple, metric_name_)
        eval_metrics_dict_[metric_name_] = metric_tuple_
    
    return eval_metrics_dict_

def check_eval_metrics(eval_metrics):
    """Check parameter `eval_metrics`."""
    if eval_metrics is None:
        return EVAL_METRICS_DEFAULT
    elif isinstance(eval_metrics, dict):
        return _check_eval_metrics_dict(eval_metrics)
    else: 
        raise TypeError(
            f"'eval_metrics' must be of type `dict`, got {type(eval_metrics)}, please check your usage."
            + EVAL_METRICS_INFO
        )


TRAIN_VERBOSE_TYPE = {
    'granularity': numbers.Integral,
    'print_distribution': bool,
    'print_metrics': bool,
}

TRAIN_VERBOSE_DEFAULT = {
    # 'granularity' will be set to int(n_estimators_ensemble/10)
    #  when check_train_verbose() is called
    'print_distribution': True,
    'print_metrics': True,
}

TRAIN_VERBOSE_DICT_INFO = \
        " When 'train_verbose' is `dict`, at least one of the following" + \
        " terms should be specified: " + \
        " {'granularity': `int` (default=1)," + \
        " 'print_distribution': `bool` (default=True)," + \
        " 'print_metrics': `bool` (default=True)}."


def check_train_verbose(train_verbose:bool or numbers.Integral or dict,
                        n_estimators_ensemble:int, training_type:str, 
                        **ignored_properties):
                        # n_estimators_ensemble:int,):

    train_verbose_ = copy(TRAIN_VERBOSE_DEFAULT)
    train_verbose_.update({
        'granularity': max(1, int(n_estimators_ensemble/10))
    })

    if training_type == 'parallel':
        # For ensemble classifiers trained in parallel
        # train_verbose can only be of type bool
        if isinstance(train_verbose, bool):
            if train_verbose == True:
                train_verbose_['print_distribution'] = False 
                return train_verbose_
            if train_verbose == False:
                return False
        else: raise TypeError(
            f"'train_verbose' can only be of type `bool`"
            f" for ensemble classifiers trained in parallel,"
            f" gor {type(train_verbose)}."
        )
    
    elif training_type == 'iterative':
        # For ensemble classifiers trained in iterative manner
        # train_verbose can be of type bool / int / dict
        if isinstance(train_verbose, bool):
            if train_verbose == True:
                return train_verbose_
            if train_verbose == False:
                return False

        if isinstance(train_verbose, numbers.Integral):
            train_verbose_.update({'granularity': train_verbose})
            return train_verbose_
            
        if isinstance(train_verbose, dict):
            # check key value type
            set_diff_verbose_keys = set(train_verbose.keys()) - set(TRAIN_VERBOSE_TYPE.keys())
            if len(set_diff_verbose_keys) > 0:
                raise ValueError(
                    f"'train_verbose' keys {set_diff_verbose_keys} are not supported." + \
                    TRAIN_VERBOSE_DICT_INFO
                )
            for key, value in train_verbose.items():
                if not isinstance(value, TRAIN_VERBOSE_TYPE[key]):
                    raise TypeError(
                        f"train_verbose['{key}'] has wrong data type, should be {TRAIN_VERBOSE_TYPE[key]}." + \
                        TRAIN_VERBOSE_DICT_INFO
                    )
            train_verbose_.update(train_verbose)
            return train_verbose_
            
        else: raise TypeError(
            f"'train_verbose' should be of type `bool`, `int`, or `dict`, got {type(train_verbose)} instead." + \
            TRAIN_VERBOSE_DICT_INFO
        )
    
    else: raise NotImplementedError(
        f"'check_train_verbose' for 'training_type' = {training_type}"
        f" needs to be implemented."
    )


def check_type(param, param_name:str, typ, typ_name:str=None):
    if not isinstance(param, typ):
        typ_name = str(typ) if typ_name is None else typ_name
        raise ValueError(
            f"'{param_name}' should be of type `{typ_name}`,"
            f" got {type(param)}."
        )
    return param

    
def check_pred_proba(y_pred_proba, n_samples, n_classes, dtype=None):
    """Private function for validating y_pred_proba"""
    if dtype is not None and dtype not in [np.float32, np.float64]:
        dtype = np.float64

    if dtype is None:
        dtype = [np.float64, np.float32]
    y_pred_proba = check_array(
        y_pred_proba, accept_sparse=False, ensure_2d=False, dtype=dtype,
        order="C"
    )
    if y_pred_proba.ndim != 2:
        raise ValueError("Predicted probabilites must be 2D array")

    if y_pred_proba.shape != (n_samples, n_classes):
        raise ValueError("y_pred_proba.shape == {}, expected {}!"
                        .format(y_pred_proba.shape, (n_samples, n_classes)))
    return y_pred_proba