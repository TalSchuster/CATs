"""
Training early and meta classifiers on top of a pre-trained Transformer for VitaminC tasks.

Install the VitaminC package from here: https://github.com/TalSchuster/VitaminC

Code modified from: https://github.com/TalSchuster/VitaminC/blob/main/scripts/fact_verification.py
"""
import logging
import os
import sys
import json
import torch
from torch import nn
from torch import optim
from dataclasses import dataclass, field
from typing import Optional, List
from sklearn.metrics import f1_score
from copy import deepcopy

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AdamW,
)

from vitaminc.processing.multitask_sent_pair_cls import (
    VitCFactVerificationProcessor,
    VitCDataTrainingArguments,
    VitCDataset,
    )

from src.modeling import AlbertWithEarlyExits, ModelArguments
from src.utils import ECELoss


logger = logging.getLogger(__name__)

def write_predictions(cls_logits, meta_logits, dataset, output_pred_file):
    pred_labels = np.argmax(cls_logits, axis=-1)
    with open(output_pred_file, "w") as writer:
        for i in range(len(cls_logits)):
            ex_logits = cls_logits[i]
            ex_pred_label = [dataset.get_labels()[p] for p in pred_labels[i]]

            # Counting consecutive similar predictions
            patience = [0] * len(ex_pred_label)
            for _ in range(len(ex_pred_label)):
                patience = [0] + [patience[i-1]+1 if ex_pred_label[i-1] == ex_pred_label[i] else 0 for i in range(1, len(ex_pred_label))]
            out_dict = {
                "ind": i,
                "layer_logits": ex_logits.tolist(),
                "layer_probs": torch.softmax(torch.tensor(ex_logits), -1).tolist(),
                "predicted_labels": ex_pred_label,
                "patience": patience,
                "gold_label_ind": dataset[i].label,
                "gold_label": dataset.get_labels()[dataset[i].label],
            }
            if meta_logits is not None:
                ex_meta_logits = meta_logits[i]
                out_dict["layer_meta_logits"] = ex_meta_logits.tolist(),
                out_dict["layer_meta_probs"] = torch.softmax(torch.tensor(ex_meta_logits),-1)[:,1].tolist(),

            writer.write(json.dumps(out_dict) + "\n")


@dataclass
class VitCTrainingArgs(TrainingArguments):
    do_test: bool = field(
        default=False, metadata={"help": "Run evaluation on test set (needs labels)"}
    )
    do_predict_eval: bool = field(
        default=False, metadata={"help": "Get prediction for evaluation set"}
    )
    scaling_iterations: int = field(
        default=None, metadata={"help": "Number of iterations for temp scaling (None->training)"}
    )
    meta_iterations: int = field(
        default=None, metadata={"help": "Number of iterations for meta training (None->training)"}
    )
    early_train_file: str = field(
        default=None, metadata={"help": "Path to file with training data for early classifiers"}
    )
    early_scaling_file: str = field(
        default=None, metadata={"help": "Path to file with data for temperature scaling"}
    )
    early_meta_file: str = field(
        default=None, metadata={"help": "Path to file with data training the meta classifier"}
    )
    meta_learning_rate: float = field(
        default=None, metadata={"help": "Learning rate for the meta training (None->learning_rate)."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, VitCDataTrainingArguments, VitCTrainingArgs))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    os.makedirs(training_args.output_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(training_args.output_dir, 'log.txt'))
    logging.getLogger("transformers").setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logging.getLogger("transformers").addHandler(fh)
    logging.root.addHandler(fh)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    if data_args.test_tasks is None:
        data_args.test_tasks = data_args.tasks_names

    # Set seed
    set_seed(training_args.seed)

    num_labels = len(VitCFactVerificationProcessor().get_labels())

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.tasks_names,
        cache_dir=model_args.cache_dir,
    )
    config.use_history_logits = model_args.use_history_logits
    config.use_early_poolers = model_args.use_early_poolers
    config.use_consistency_loss = model_args.use_consistency_loss
    config.use_meta_predictors = model_args.use_meta_predictors
    config.joint_meta = model_args.joint_meta
    config.shared_meta = model_args.shared_meta
    config.early_pooler_hidden_size = model_args.early_pooler_hidden_size
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    model = AlbertWithEarlyExits.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        VitCDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, file_path=training_args.early_train_file) if training_args.do_train else None
    )
    scaling_dataset = (
        VitCDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, file_path=training_args.early_scaling_file) if training_args.early_scaling_file else None
    )
    meta_dataset = (
        VitCDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, file_path=training_args.early_meta_file) if training_args.early_meta_file else None
    )
    eval_dataset = (
        VitCDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval or training_args.do_predict_eval else None
    )

    def compute_metrics_fn(p: EvalPrediction):
        meta_logits = None
        ece_measure = ECELoss()
        if type(p.predictions) is tuple:
            cls_logits = p.predictions[0]
            meta_logits = p.predictions[1]
        else:
            cls_logits = p.predictions
        top_preds = np.argmax(cls_logits[:,-1,:], axis=-1)
        acc = (top_preds == p.label_ids).mean()
        ece = ece_measure(torch.Tensor(cls_logits[:,-1,:]), torch.Tensor(p.label_ids))
        metrics = {"accuracy": acc, "ece": ece.item()}
        for i in range(cls_logits.shape[1] - 1):
            preds = np.argmax(cls_logits[:,i,:], axis=-1)
            acc = (preds == p.label_ids).astype(np.float32).mean().item()
            ece = ece_measure(torch.Tensor(cls_logits[:,i,:]), torch.Tensor(p.label_ids))
            metrics[f"ece_{i}"] = ece.item()
            metrics[f"accuracy_{i}"] = acc
            consistency = (preds == top_preds).mean()
            metrics[f"consistency_{i}"] = consistency
            if meta_logits is not None:
                meta_labels = np.equal(preds, top_preds).astype(int)
                meta_preds = np.argmax(meta_logits[:,i,:], axis=-1)
                meta_acc = (meta_preds == meta_labels).mean()
                metrics[f"meta_accuracy_{i}"] = meta_acc

        return metrics

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn
    )

    # Training
    if training_args.do_train:
        train_results = trainer.train()
        metrics = train_results.metrics

        if training_args.early_scaling_file:
            trainer.save_model(os.path.join(training_args.output_dir, "pre_scaling"))
            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(os.path.join(training_args.output_dir, "pre_scaling"))
            if training_args.do_eval:
                logger.info("*** Evaluate before scaling***")

                tasks_str = "-".join(eval_dataset.args.tasks_names)
                output_eval_file = os.path.join(
                    training_args.output_dir, "pre_scaling", f"eval_results_{tasks_str}.txt"
                )
                eval_result = trainer.evaluate(eval_dataset=eval_dataset)

                if trainer.is_world_process_zero():
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Eval results before scaling {} *****".format(tasks_str))
                        for key, value in eval_result.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))

        else:
            trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Temperature scaling
    if training_args.early_scaling_file:
        logger.info("*** Scaling temperatures ***")
        model.temp_scaling_mode = True
        optimizer = AdamW([model.early_temperatures], lr=0.001)
        scaling_args = deepcopy(training_args)
        if training_args.scaling_iterations is not None:
            scaling_args.max_steps = training_args.scaling_iterations
            scaling_args.save_steps = training_args.scaling_iterations
            scaling_args.fp16 = False
        trainer = Trainer(
            model=model,
            args=scaling_args,
            optimizers=(optimizer, None),
            train_dataset=scaling_dataset,
            compute_metrics=compute_metrics_fn
        )

        train_results = trainer.train()
        metrics = train_results.metrics
        logger.info(metrics)

        trainer.save_model()

    # Train the meta predictors
    if training_args.early_meta_file:
        logger.info("*** Training meta predictors ***")
        assert model.use_meta_predictors
        model.temp_scaling_mode = False
        model.meta_training_mode = True
        meta_args = deepcopy(training_args)
        if training_args.scaling_iterations is not None:
            meta_args.max_steps = training_args.meta_iterations
            meta_args.save_steps = training_args.meta_iterations
        if training_args.meta_learning_rate is not None:
            meta_args.learning_rate = training_args.meta_learning_rate
        trainer = Trainer(
            model=model,
            args=meta_args,
            train_dataset=meta_dataset,
            compute_metrics=compute_metrics_fn
        )

        train_results = trainer.train()
        metrics = train_results.metrics
        logger.info(metrics)

        trainer.save_model()


    if model.use_meta_predictors:
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics_fn
        )
        model.meta_eval_mode = True

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        tasks_str = "-".join(eval_dataset.args.tasks_names)
        output_eval_file = os.path.join(
            training_args.output_dir, f"eval_results_{tasks_str}.txt"
        )
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(tasks_str))
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        eval_results.update(eval_result)

    if training_args.do_predict_eval:
        tasks_str = "-".join(eval_dataset.args.tasks_names)
        logger.info("Predicting for %s validation", tasks_str)
        predictions = trainer.predict(test_dataset=eval_dataset).predictions
        meta_logits = None
        if type(predictions) is tuple:
            cls_logits = predictions[0]
            meta_logits = predictions[1]
        else:
            cls_logits = predictions[0]
        output_pred_file = os.path.join(
            training_args.output_dir, f"eval_preds_{tasks_str}.jsonl"
        )
        write_predictions(cls_logits, meta_logits, eval_dataset, output_pred_file)

    if training_args.do_test or training_args.do_predict:
        test_args = deepcopy(data_args)
        test_args.dataset_size = None
        test_args.tasks_ratios = [1.]
        for task_name in data_args.test_tasks:
            test_args = deepcopy(test_args)
            test_args.tasks_names = [task_name]
            test_dataset = VitCDataset(
                                test_args,
                                tokenizer=tokenizer,
                                mode="test",
                                cache_dir=model_args.cache_dir
            )
            if training_args.do_predict:
                logger.info("Predicting for %s test", task_name)
                predictions = trainer.predict(test_dataset=test_dataset).predictions
                meta_logits = None
                if type(predictions) is tuple:
                    cls_logits = predictions[0]
                    meta_logits = predictions[1]
                else:
                    cls_logits = predictions[0]
            if training_args.do_test:
                # TODO: Use predictions from do_predict for evaluation.
                logger.info("Evaluating on %s", test_dataset.args.tasks_names[0])
                eval_result = trainer.evaluate(eval_dataset=test_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"test_results_{task_name}.txt"
            )
            output_pred_file = os.path.join(
                training_args.output_dir, f"test_preds_{task_name}.jsonl"
            )
            if trainer.is_world_process_zero():
                if training_args.do_test:
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Test results {} *****".format(test_dataset.args.tasks_names[0]))
                        for key, value in eval_result.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))

                if training_args.do_predict:
                    write_predictions(cls_logits, meta_logits, test_dataset, output_pred_file)

    return eval_results


if __name__ == "__main__":
    main()
