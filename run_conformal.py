"""Run experiments for conformal retrospective analysis."""

import argparse
import json
import numpy as np
import os
from pprint import pprint
from datetime import datetime

from src.conformal import evaluate_trials


def load_data(dataset_file, nonconformity, regression=False, reg_tolerance=0.5):
    """Load data and sample into trials."""
    examples = []
    with open(dataset_file, "r") as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)

    print("Number of examples: %d" % len(examples))

    num_examples = len(examples)
    num_layers = len(examples[0]["layer_logits"])
    data_x = np.empty((num_examples, num_layers - 1))
    data_y = np.zeros((num_examples, num_layers - 1))
    data_gold = np.zeros((num_examples, num_layers))
    top_acc = np.zeros(num_examples)

    for i, example in enumerate(examples):
        for l in range(num_layers - 1):
            if nonconformity == "meta":
                score = example["layer_meta_probs"][0][l]
            elif nonconformity == "max_prob":
                score = max(example["layer_probs"][l])
            elif nonconformity == "entropy":
                score = (example["layer_probs"][l] * np.log(example["layer_probs"][l])).sum()
            elif nonconformity == "max_diff":
                top = sorted(example["layer_probs"][l], reverse=True)[:2]
                score = top[0] - top[1]
            elif nonconformity == "random":
                score = np.random.random()
            else:
                raise NotImplementedError
            data_x[i, l] = score
            if regression:
                data_y[i, l] = int(abs(example["layer_logits"][l][0] - example["layer_logits"][-1][0]) <= reg_tolerance)
                data_gold[i, l] = int(abs(example["layer_logits"][l][0] - example["gold_label"]) <= reg_tolerance)
            else:
                data_y[i, l] = int(example["predicted_labels"][l] == example["predicted_labels"][-1])
                data_gold[i, l] = int(example["predicted_labels"][l] == example["gold_label"])
        if regression:
            data_gold[i, -1] = int(abs(example["layer_logits"][-1][0] - example["gold_label"]) <= reg_tolerance)
            top_acc[i] = int(abs(example["layer_logits"][-1][0] - example["gold_label"]) <= reg_tolerance)
        else:
            data_gold[i, -1] = int(example["predicted_labels"][-1] == example["gold_label"])
            top_acc[i] = int(example["predicted_labels"][-1] == example["gold_label"])

    return data_x, data_y, data_gold, np.mean(top_acc)


def create_trials(num_examples, num_trials):
    """Create random sample of trials."""
    trials = []
    for _ in range(num_trials):
        idx = np.random.permutation(num_examples).tolist()
        cal = idx[:int(0.8 * len(idx))]
        test = idx[int(0.8 * len(idx)):]
        trials.append((cal, test))
    return trials


def main(args):
    np.random.seed(42)
    assert all([x in [0,1] for x in args.conditional])

    if not args.trials_file:
        args.trials_file = os.path.splitext(args.dataset_file)[0]
        args.trials_file += ("-trials=%d.json" % (args.num_trials))
        os.makedirs(os.path.dirname(args.trials_file), exist_ok=True)

    if not args.output_file:
        args.output_file = os.path.splitext(args.dataset_file)[0]
        args.output_file += ("-trials=%d-results.jsonl" % (args.num_trials))

    if args.overwrite_results:
        open(args.output_file, "w").close()

    if os.path.exists(args.trials_file) and not args.overwrite_trials:
        print("Loading trials from %s" % args.trials_file)
        with open(args.trials_file, "r") as f:
            trials = json.load(f)
    else:
        print("Loading data...")
        data_x, data_y, data_gold, top_acc = load_data(args.dataset_file, "random", regression=args.regression)
        trials = create_trials(len(data_x), args.num_trials)
        print("Writing trials to %s" % args.trials_file)
        with open(args.trials_file, "w") as f:
            json.dump(trials, f)

    print("Will write to %s" % args.output_file)
    for epsilon in args.epsilons:
        static_done = False
        res_dict = {}
        for method in args.methods:
            res_dict[method] = {}
            for nonconformity in args.nonconformities:
                res_dict[method][nonconformity] = {}
                for conditional in args.conditional:
                    conditional = conditional == 1
                    if "naive" in method:
                        if nonconformity not in ["max_prob", "meta"]:
                            continue
                    if method == "static":
                        # Run static only once cuz it will be the same...
                        if static_done:
                            continue
                        static_done = True

                    print("Loading data...")
                    data_x, data_y, data_gold, top_acc = load_data(args.dataset_file, nonconformity, regression=args.regression, reg_tolerance=args.reg_tolerance)
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    print(f"{current_time}: Running eps={epsilon}, method={method}, nonconformity={nonconformity}, conditional={conditional}")

                    res_dict[method][nonconformity][conditional] = evaluate_trials(
                        trials=trials,
                        scores=data_x,
                        labels=data_y,
                        golds=data_gold,
                        epsilon=epsilon,
                        conditioned=conditional,
                        top_acc=top_acc,
                        method=method,
                        threads=args.threads)

        with open(args.output_file, "a") as f:
            output = {"epsilon": epsilon,
                      "results": res_dict,
                      "top_acc": 100*top_acc,
                      "lower_target_acc": 100*(top_acc * (1-epsilon)),
                }
            f.write(json.dumps(output) + "\n")

        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, default="results/cat_model/eval_preds.jsonl")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--trials_file", type=str, default=None)
    parser.add_argument("--num_trials", type=int, default=25)
    parser.add_argument("--overwrite_trials", action="store_true")
    parser.add_argument("--overwrite_results", action="store_true")
    parser.add_argument("--conditional", nargs="+", type=int, default=[0], help="1=True, 0=False")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.1])
    parser.add_argument("--threads", type=int, default=25)
    parser.add_argument("--methods", type=str, nargs="+", default=["shared"])
    parser.add_argument("--nonconformities", nargs="+", type=str, default=["meta"])
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--reg_tolerance", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
