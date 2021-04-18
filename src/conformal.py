"""Conformal prediction library for early exits."""

import multiprocessing
import functools
import tqdm
import numpy as np

INF = 1e8


def adaptive_quantile(cal_x, cal_y, epsilon):
    # Steps: regress quantile using first layer scores.
    # Recompute nonconformity scores.
    pass


def early_exit(x, cal_x, cal_y, epsilon, method="naive"):
    """Early exit prediction.

    All nonconformity scores are for f_k(x) != f_L(x).

    Args:
        x: Nonconformity scores of size [num_layers].
        cal_x: Nonconformity scores of size [num_calibration, num_layers].
        cal_y: Ground truth labels of size [num_calibration].
        method: Calibration method to use.

    Returns:
        test_y: Layer indices for prediction of first early exit layer.
    """
    if method == "none":
        return indep_early_exit(x, cal_x, cal_y, epsilon, correction="none")
    if method == "indep":
        return indep_early_exit(x, cal_x, cal_y, epsilon, correction="bonferroni")
    elif method == "shared":
        return shared_early_exit(x, cal_x, cal_y, epsilon)
    elif method == "naive":
        return naive_early_exit(x, cal_x, cal_y, epsilon, correction="none")
    elif method == "naive_bonferroni":
        return naive_early_exit(x, cal_x, cal_y, epsilon, correction="bonferroni")
    elif method == "static":
        return static_early_exit(x, cal_x, cal_y, epsilon)
    else:
        raise NotImplementedError


def naive_early_exit(x, cal_x, cal_y, epsilon, correction="none"):
    """
        Naive method where we exit by a predefined threshold without conformalization.
        This method ignores the calibration set.
        """

    num_layers = len(x)
    if correction == "bonferroni":
        epsilon /= num_layers

    # Iterate layers. Break if confident.
    for l in range(num_layers):
        if x[l] >= 1 - epsilon:
            return l

    return num_layers


def static_early_exit(x, cal_x, cal_y, epsilon, correction="none"):
    """
        Exit on the first layer that reaches 1 - epsilon accuracy on the calibration set.
        This ignores the value of x.
    """

    num_layers = len(x)
    valid_layers = np.where(cal_y.mean(0) >= 1 - epsilon)[0]
    if len(valid_layers) > 0:
        return min(valid_layers)
    else:
        return num_layers


def indep_early_exit(x, cal_x, cal_y, epsilon, correction="bonferroni"):
    """Eeach layer is individually conformalized."""

    def _compute_pvalue(value, values):
        greater = np.sum(values > value)
        equal = np.sum(values == value)
        equal = np.random.random() * equal
        pvalue = (greater + equal + 1) / (len(values) + 1)
        return pvalue.item()

    num_layers = len(x)
    if correction == "bonferroni":
        epsilon /= num_layers

    # Iterate layers. Break if confident.
    for l in range(num_layers):
        cal_xl = cal_x[:, l][cal_y[:, l] == 0]
        pvalue = _compute_pvalue(x[l], cal_xl)
        if pvalue < epsilon:
            return l

    return num_layers


def shared_early_exit(x, cal_x, cal_y, epsilon):
    """Direct inner-outer method of Cauchois et. al."""
    # For each point, take *maximum* nonconformity score for layers
    # that are *not* equal to the final layer (i.e., y = 0).
    valid = np.any(cal_y == 0, axis=-1)
    cal_x = cal_x[valid]
    cal_y = cal_y[valid]
    cal_x = cal_x - cal_y * INF
    cal_x = cal_x.max(axis=-1)

    # Compute quantile.
    cal_x = np.concatenate([cal_x, [np.inf]])
    quantile = np.quantile(cal_x, max(0, 1 - epsilon), interpolation="higher")

    # Iterate layers. Break if confident.
    num_layers = len(x)
    for l in range(num_layers):
        if x[l] > quantile:
            return l

    return num_layers


def evaluate_trial(trial, epsilon, method="naive", conditioned=False, top_acc=1., unaggregated=False):
    """
    Test batch of examples with fixed calibration set.
    conditioned: calibrate only on examples that f_L(x) == corect y
    
    data_x: nonconformity scores
    data_y: consistency with last layer [0/1] * num_layers
    data_gold: correct prediction [0/1] * num_layers
    """
    global data_x, data_y, data_gold

    cal_idx, test_idx = trial
    if conditioned:
        cal_x = np.vstack([data_x[i] for i in cal_idx if data_gold[i][-1] == 1])
        cal_y = np.vstack([data_y[i] for i in cal_idx if data_gold[i][-1] == 1])
    else:
        cal_x = np.vstack([data_x[i] for i in cal_idx])
        cal_y = np.vstack([data_y[i] for i in cal_idx])
    test_x = np.vstack([data_x[i] for i in test_idx])
    test_y = np.vstack([data_y[i] for i in test_idx])
    gold_y = np.vstack([data_gold[i] for i in test_idx])

    layers = []
    consistent = []
    conditioned_consistent = []
    correct = []
    for x, y, gold in zip(test_x, test_y, gold_y):
        layer = early_exit(x, cal_x, cal_y, epsilon, method)
        layers.append(layer)
        if layer == len(y):
            # Last layer
            consistent.append(1)
        else:
            consistent.append(y[layer])
        correct.append(gold[layer])
        if gold[-1] == 1:
            if layer == len(y):
                conditioned_consistent.append(1)
            else:
                conditioned_consistent.append(y[layer])

    layers_dist = np.bincount(layers,minlength=len(y)+1) / len(layers)

    if unaggregated:
        return layers, consistent, correct
    else:
        return (np.mean(layers),
           np.mean(consistent),
           np.mean(correct),
           np.mean(conditioned_consistent),
           layers_dist)


def init(scores, labels, golds):
    global data_x, data_y, data_gold
    data_x = scores
    data_y = labels
    data_gold = golds


def evaluate_trials(trials, scores, labels, golds, epsilon, method, conditioned, top_acc, threads=1):
    """Evaluate conformalized early exit.

    Args:
        trials: List of (calibration idx, test_idx).
        scores: Array of nonconformity scores [num_examples, num_layers].
        labels: Array of labels (if f_k(x) == f_L(x)).
        golds: Array of labels (if f_k(x) == gold y).
        method: shared, indep etc.
        conditioned: If to calibrate only on examples where f_L(x) == gold y.
        top_acc: accuracy of f_L(x) (for adjusting the conditioned calibration)
        threads: Number of multiprocessing workers to use.

    Returns:
        result: Average layer exit and consistency.
    """
    # Only use multiprocessing with threads > 1.
    if threads > 1:
        workers = multiprocessing.Pool(threads, initializer=init, initargs=(scores, labels, golds))
        map_fn = workers.imap_unordered
    else:
        init(scores, labels, golds)
        map_fn = map

    worker_fn = functools.partial(
        evaluate_trial,
        epsilon=epsilon,
        method=method,
        conditioned=conditioned,
        top_acc=top_acc)

    layers = []
    consistencies = []
    accuracies = []
    cond_consistencies = []
    layer_dists = []
    with tqdm.tqdm(total=len(trials)) as pbar:
        for layer, const, acc, cond_const, layer_dist in map_fn(worker_fn, trials):
            layers.append(layer)
            consistencies.append(100*const)
            accuracies.append(100*acc)
            cond_consistencies.append(100*cond_const)
            layer_dists.append(layer_dist)

    def measures(results):
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0)
        p84 = np.percentile(results, 84, axis=0)
        p16 = np.percentile(results, 16, axis=0)

        if isinstance(mean, np.ndarray):
            mean = list(mean)
            std = list(std)
            p84 = list(p84)
            p16 = list(p16)

        return [mean, std, p84, p16]

    return dict(layer=measures(layers), consistency=measures(consistencies), acc=measures(accuracies), cond_consistency=measures(cond_consistencies), layer_dists=measures(layer_dists))
