import torch
import numpy as np
from torch import Tensor, device, dtype, nn
from torch.nn import CrossEntropyLoss, MSELoss
from dataclasses import dataclass, field
from typing import Optional, List

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    AlbertForSequenceClassification,
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    use_history_logits: bool = field(
        default=False, metadata={"help": "Use the logits from lower layers as features"}
    )
    use_early_poolers: bool = field(
        default=False, metadata={"help": "Use one nonlinear layer before the early classifier (like used for the top cls)"}
    )
    use_consistency_loss: bool = field(
        default=False, metadata={"help": "Train to immitate last layer's prediction instead of gold label"}
    )
    use_meta_predictors: bool = field(
        default=False, metadata={"help": "Add meta early exit classifier that predicts if the prediction "
                                         "of the early exit classifier is correct"}
    )
    joint_meta: bool = field(
        default=False, metadata={"help": "Train meta classifier to also predict success of top layer"}
    )
    shared_meta: bool = field(
        default=False, metadata={"help": "Use a shared meta classifier for all layers. Using general features"}
    )
    early_pooler_hidden_size: int = field(
        default=32, metadata={"help": "Size of hidden layer for early poolers"}
    )
    regression_tolerance: float = field(
        default=0.5, metadata={"help": "How much tolerance can we afford for the prediction."}
    )


@dataclass
class EarlyExitSequenceClassifierOutput(SequenceClassifierOutput):
    meta_logits: torch.FloatTensor = None


class AlbertWithEarlyExits(AlbertForSequenceClassification):
    """
    Wraps Transformer models for sequence classification and adds per layer early
    classifiers. The wrapped Transformer should be pretrained for the task and
    fixed during the fine-tuning here (to preserve performance of top
    classifier).

    This wrapper doesn't exit early in practice as it wraps the original Transformer.
    It could be used for training the intermediate classifiers, calibrating confidence
    thresholds and for retrospective evaluation. To use the resultant values for early
    inference, we'll need to modify the underlying Transformer code.
    """
    def __init__(self, config):
        super().__init__(config)

        # Don't include last layer.
        self.num_inner_classifiers = config.num_hidden_layers - 1

        self.use_history_logits = config.use_history_logits
        self.use_early_poolers = config.use_early_poolers
        self.use_consistency_loss = config.use_consistency_loss
        self.use_meta_predictors = config.use_meta_predictors
        self.joint_meta = config.joint_meta
        self.shared_meta = config.shared_meta
        self.regression_tolerance = config.regression_tolerance
        if self.use_early_poolers:
            if self.use_history_logits:
                self.early_poolers = nn.ModuleList([nn.Linear(config.hidden_size + i*self.num_labels,
                                                              config.early_pooler_hidden_size)
                                                    for i in range(self.num_inner_classifiers)])
            else:
                self.early_poolers = nn.ModuleList([nn.Linear(config.hidden_size,
                                                              config.early_pooler_hidden_size)
                                                    for _ in range(self.num_inner_classifiers)])

            self.early_classifiers = nn.ModuleList([nn.Linear(config.early_pooler_hidden_size,
                                                              config.num_labels)
                                            for i in range(self.num_inner_classifiers)])
        else:
            if self.use_history_logits:
                self.early_classifiers = nn.ModuleList([nn.Linear(config.hidden_size +i*config.num_labels,
                                                                  config.num_labels)
                                            for i in range(self.num_inner_classifiers)])
            else:
                self.early_classifiers = nn.ModuleList([nn.Linear(config.hidden_size, config.num_labels)
                                            for _ in range(self.num_inner_classifiers)])

        if self.use_meta_predictors:
            if self.shared_meta:
                if self.num_labels == 1:
                    self.meta_pooler = nn.Linear(config.early_pooler_hidden_size + (self.num_inner_classifiers - 1),
                                                     config.early_pooler_hidden_size)
                else:
                    self.meta_pooler = nn.Linear(config.num_labels + config.early_pooler_hidden_size + 2 + (self.num_inner_classifiers - 1),
                                                     config.early_pooler_hidden_size)
                self.meta_predictor = nn.Linear(config.early_pooler_hidden_size, 2)
            else:
                if self.num_labels == 1:
                    self.meta_poolers = nn.ModuleList([nn.Linear(config.early_pooler_hidden_size + (self.num_inner_classifiers - 1),
                                                    config.early_pooler_hidden_size)
                                                   for _ in range(self.num_inner_classifiers)])
                else:
                    self.meta_poolers = nn.ModuleList([nn.Linear(config.num_labels + config.early_pooler_hidden_size + 2 + (self.num_inner_classifiers - 1),
                                                    config.early_pooler_hidden_size)
                                                   for _ in range(self.num_inner_classifiers)])

                self.meta_predictors = nn.ModuleList([nn.Linear(config.early_pooler_hidden_size, 2)
                                                      for _ in range(self.num_inner_classifiers)])

        self.early_temperatures = nn.Parameter(torch.ones(self.num_inner_classifiers))
        self.early_pooler_activation = nn.Tanh()
        self.early_dropout = nn.Dropout(config.classifier_dropout_prob)

        self.temp_scaling_mode = False
        self.meta_training_mode = False
        self.meta_eval_mode = False

    def run_early_classifiers(self, top_logits, outputs):
        all_logits = top_logits.new_zeros(
                                top_logits.shape[0],  # Batch size
                                self.num_inner_classifiers + 1,
                                self.num_labels)
        all_logits[:,-1,:] = top_logits

        hidden_cls = top_logits.new_zeros(
                                top_logits.shape[0],  # Batch size
                                self.num_inner_classifiers,
                                self.config.early_pooler_hidden_size)

        # Intermediate classifiers
        for i in range(self.num_inner_classifiers):
            if not self.temp_scaling_mode:
                cls = outputs[2][i+1][:,0,:]
                if self.use_history_logits and i > 0:
                    cls = torch.cat((cls, all_logits[:,:i,:].view(-1, i*self.num_labels)), dim=-1)
                cls = self.early_dropout(cls)
                if self.use_early_poolers:
                    cls = self.early_poolers[i](cls)
                    cls = self.early_pooler_activation(cls)
                layer_logits = self.early_classifiers[i](cls)
                with torch.no_grad():
                    layer_logits /= self.early_temperatures[i]
            else:
                with torch.no_grad():
                    cls = outputs[2][i+1][:,0,:]
                    if self.use_history_logits and i > 0:
                        cls = torch.cat((cls, all_logits[:,:i,:].view(-1, i*self.num_labels)), dim=-1)
                    cls = self.early_dropout(cls)
                    if self.use_early_poolers:
                        cls = self.early_poolers[i](cls)
                        cls = self.early_pooler_activation(cls)
                    layer_logits = self.early_classifiers[i](cls)
                layer_logits /= self.early_temperatures[i]
            all_logits[:,i,:] = layer_logits
            hidden_cls[:,i,:] = cls

        return all_logits, hidden_cls

    def run_meta_predictors(self, all_logits, hidden_cls):
        meta_logits = all_logits.new_zeros(
                                all_logits.shape[0],  # Batch size
                                self.num_inner_classifiers,
                                2)
        all_probs = torch.softmax(all_logits, -1)
        for i in range(self.num_inner_classifiers):

            # [batch_size, hidden_size]
            cls = hidden_cls[:,i,:]
            cls = self.early_dropout(cls)

            # [batch_size, num_labels]
            probs = all_probs[:,i,:]

            progress = all_logits.new_ones(all_logits.shape[0], 1) * i / self.num_inner_classifiers
            history = all_logits.new_zeros(all_logits.shape[0], self.num_inner_classifiers - 2)

            if self.num_labels == 1:
                #  We are doing regression.
                if i > 1:
                    history[:,:i-1] = all_logits[:,:i-1,0]
                cls = torch.cat((cls, history, progress), dim=-1)
            else:
                top_two, arg_top_two = probs.topk(2, dim=-1)
                max_diff = (top_two[:,0] - top_two[:,1]).unsqueeze(-1)
                max_prob = top_two[:,:1]
                arg_max = arg_top_two[:,0]

                # Collects the probabilities of lower layers of the same label.
                # Masks with zeros for not yet seen layers.
                if i > 1:
                    # Don't include first layer because it's usually pretty random.
                    history[:,:i-1] = all_probs[torch.arange(all_probs.shape[0]),1:i,arg_max].flip(-1)

                pred_class = all_logits.new_zeros(all_logits.shape[0], self.num_labels)
                pred_class[torch.arange(all_probs.shape[0]), arg_max] = 1

                cls = torch.cat((pred_class, cls, max_prob, max_diff, history, progress), dim=-1)

            if self.shared_meta:
                # [batch_size, early_pooler_hidden_size]
                cls = self.meta_pooler(cls)
                cls = self.early_pooler_activation(cls)

                # [batch_size, 2]
                preds = self.meta_predictor(cls)
            else:
                cls = self.meta_poolers[i](cls)
                cls = self.early_pooler_activation(cls)

                # [batch_size, 2]
                preds = self.meta_predictors[i](cls)

            meta_logits[:,i,:] = preds

        return meta_logits

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.albert.eval()

        with torch.no_grad():
            outputs = self.albert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True,  # We will need them here...
                return_dict=return_dict,
            )

            # Top classifier
            pooled_output = outputs[1]
            top_logits = self.classifier(pooled_output)

        meta_logits = None
        if self.meta_training_mode or self.meta_eval_mode:
            with torch.no_grad():
                all_logits, hidden_cls = self.run_early_classifiers(top_logits, outputs)
            meta_logits = self.run_meta_predictors(all_logits, hidden_cls)
        else:
            all_logits, _ = self.run_early_classifiers(top_logits, outputs)

        loss = None
        if not self.meta_training_mode and (labels is not None or self.use_consistency_loss):
            loss = 0.0
            #  Average over all intermediate losses
            # TODO: use view to avoid the for loop
            for i in range(self.num_inner_classifiers):
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    if self.use_consistency_loss:
                        loss += loss_fct(all_logits[:,i,:].view(-1), all_logits[:,-1,:].view(-1))
                    else:
                        loss += loss_fct(all_logits[:,i,:].view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    if self.use_consistency_loss:
                        loss += loss_fct(all_logits[:,i,:].view(-1, self.num_labels), all_logits[:,-1,:].argmax(-1).view(-1))
                    else:
                        loss += loss_fct(all_logits[:,i,:].view(-1, self.num_labels), labels.view(-1))

            loss /= self.num_inner_classifiers

        if self.meta_training_mode:
            loss = 0.0
            top_correct = all_logits[:,-1,:].argmax(-1).view(-1).eq(labels.view(-1)).long()
            for i in range(self.num_inner_classifiers):
                if self.num_labels == 1:
                    #  We are doing regression
                    layer_preds = all_logits[:,i,:].view(-1)
                    meta_labels = torch.abs(layer_preds - all_logits[:,-1,:].view(-1)).le(self.regression_tolerance).long()
                else:
                    # Create labels by whether the early prediction is consistant or not.
                    layer_preds = all_logits[:,i,:].argmax(-1).view(-1)
                    meta_labels = layer_preds.eq(all_logits[:,-1,:].argmax(-1).view(-1)).long()
                    if self.joint_meta:
                        # Label is 1 iff layer i == top layer AND top layer is correct.
                        meta_labels = meta_labels * top_correct

                if len(meta_labels.bincount()) == 2:
                    label_weights = len(meta_labels) / meta_labels.bincount().float()
                else:
                    # All labels are same
                    label_weights = None
                loss_fct = CrossEntropyLoss(weight=label_weights)
                loss += loss_fct(meta_logits[:,i,:].view(-1, 2), meta_labels.view(-1))

            loss /= self.num_inner_classifiers

        if not return_dict:
            if output_hidden_states:
                output = (all_logits,) + outputs[2:]
            else:
                # Correct the override of output_hidden_states
                output = (all_logits,) + outputs[3:]
            return ((loss,) + output) if loss is not None else output

        return EarlyExitSequenceClassifierOutput(
            loss=loss,
            logits=all_logits,
            meta_logits=meta_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
