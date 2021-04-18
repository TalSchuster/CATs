#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training early and meta classifiers on top of a pre-trained Transformer.

Modified from: https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""

import logging
import os
import random
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    AdamW,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from src.modeling import AlbertWithEarlyExits, ModelArguments
from src.utils import write_predictions, ECELoss


@dataclass
class EarlyTrainingArguments(TrainingArguments):
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
    meta_learning_rate: float = field(
        default=None, metadata={"help": "Learning rate for the meta training (None->learning_rate)."}
    )



# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
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
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.early_train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.early_train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`early_train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `early_train_file`."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EarlyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    os.makedirs(training_args.output_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(training_args.output_dir, 'log.txt'))
    logging.getLogger("transformers").setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logging.getLogger("transformers").addHandler(fh)
    logging.root.addHandler(fh)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {
                  "train": data_args.early_train_file,
                  "validation": data_args.validation_file,
                 }
    if data_args.early_scaling_file:
        data_files["scale"] = data_args.early_scaling_file

    if data_args.early_meta_file:
        data_files["meta"] = data_args.early_meta_file

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.
    if training_args.do_predict or training_args.do_test:
        if data_args.test_file is not None:
            train_extension = data_args.early_train_file.split(".")[-1]
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `early_train_file`."
            data_files["test"] = data_args.test_file
        else:
            raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    if data_args.early_train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset("csv", data_files=data_files)
    else:
        # Loading a dataset from local json files
        datasets = load_dataset("json", data_files=data_files, field="data")
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.use_history_logits = model_args.use_history_logits
    config.use_early_poolers = model_args.use_early_poolers
    config.use_consistency_loss = model_args.use_consistency_loss
    config.use_meta_predictors = model_args.use_meta_predictors
    config.joint_meta = model_args.joint_meta
    config.shared_meta = model_args.shared_meta
    config.early_pooler_hidden_size = model_args.early_pooler_hidden_size
    config.regression_tolerance = model_args.regression_tolerance
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AlbertWithEarlyExits.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics_fn(p: EvalPrediction):
        meta_logits = None
        ece_measure = ECELoss()
        if type(p.predictions) is tuple:
            cls_logits = p.predictions[0]
            meta_logits = p.predictions[1]
        else:
            cls_logits = p.predictions
        top_preds = np.squeeze(cls_logits[:,-1,:]) if is_regression else np.argmax(cls_logits[:,-1,:], axis=1)

        ece = ece_measure(torch.Tensor(cls_logits[:,-1,:]), torch.Tensor(p.label_ids))
        metrics = {"ece": ece.item()}

        result = metric.compute(predictions=top_preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        for key, value in result.items():
            metrics[key] = value

        for i in range(cls_logits.shape[1] - 1):
            preds = np.squeeze(cls_logits[:,i,:]) if is_regression else np.argmax(cls_logits[:,i,:], axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            for key, value in result.items():
                metrics[f"{key}_{i}"] = value

            ece = ece_measure(torch.Tensor(cls_logits[:,i,:]), torch.Tensor(p.label_ids))
            metrics[f"ece_{i}"] = ece.item()
            consistency = (preds == top_preds).mean()
            metrics[f"consistency_{i}"] = consistency
            if meta_logits is not None:
                meta_labels = np.equal(preds, top_preds).astype(int)
                meta_preds = np.argmax(meta_logits[:,i,:], axis=-1)
                meta_acc = (meta_preds == meta_labels).mean()
                metrics[f"meta_accuracy_{i}"] = meta_acc

        return metrics

    def compute_metrics(p: EvalPrediction):
        # TODO: copy metric (especially regression to other function)
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        if data_args.early_scaling_file:
            trainer.save_model(os.path.join(training_args.output_dir, "pre_scaling"))
            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(os.path.join(training_args.output_dir, "pre_scaling"))
            if training_args.do_eval:
                logger.info("*** Evaluate before scaling***")

                output_eval_file = os.path.join(
                    training_args.output_dir, "pre_scaling", f"eval_results.txt"
                )
                eval_result = trainer.evaluate(eval_dataset=eval_dataset)

                if trainer.is_world_process_zero():
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Eval results before scaling *****")
                        for key, value in eval_result.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))
        else:
            trainer.save_model()  # Saves the tokenizer too for easy upload
            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(training_args.output_dir)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Temperature scaling
    if data_args.early_scaling_file:
        logger.info("*** Scaling temperatures ***")
        scaling_dataset = datasets["scale"]
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
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Train the meta predictors
    if data_args.early_meta_file:
        logger.info("*** Training meta predictors ***")
        meta_dataset = datasets["meta"]
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
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if model.use_meta_predictors:
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics_fn
        )
        model.meta_eval_mode = True


    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict_eval:
        logger.info("Predicting for validation")
        predictions = trainer.predict(test_dataset=eval_dataset).predictions
        meta_logits = None
        if type(predictions) is tuple:
            cls_logits = predictions[0]
            meta_logits = predictions[1]
        else:
            cls_logits = predictions[0]
        output_pred_file = os.path.join(
            training_args.output_dir, f"eval_preds.jsonl"
        )
        gold_labels = [ex["label"] for ex in eval_dataset]
        write_predictions(cls_logits, meta_logits, model.config.id2label, gold_labels, output_pred_file)


    if training_args.do_test or training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            # Let's assume we have gold lablels for test for now.
            #test_dataset.remove_columns_("label")
            if training_args.do_predict:
                logger.info("Predicting for %s test", task)
                predictions = trainer.predict(test_dataset=test_dataset).predictions
                #predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
                meta_logits = None
                if type(predictions) is tuple:
                    cls_logits = predictions[0]
                    meta_logits = predictions[1]
                else:
                    cls_logits = predictions[0]

                output_pred_file = os.path.join(
                    training_args.output_dir, f"test_preds_{task}.jsonl"
                )
                gold_labels = [ex["label"] for ex in test_dataset]
                write_predictions(cls_logits, meta_logits, model.config.id2label, gold_labels, output_pred_file)


            if training_args.do_test:
                # TODO: Use predictions from do_predict for evaluation.
                logger.info("Evaluating on %s", task)
                eval_result = trainer.evaluate(eval_dataset=test_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"test_results_{task}.txt"
            )
            if trainer.is_world_process_zero():
                if training_args.do_test:
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Test results {} *****".format(task))
                        for key, value in eval_result.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
