import json
import torch
from torch import nn
import numpy as np


def write_predictions(cls_logits, meta_logits, id2label, gold_labels, output_pred_file):
    pred_labels = np.argmax(cls_logits, axis=-1)
    with open(output_pred_file, "w") as writer:
        for i in range(len(cls_logits)):
            ex_logits = cls_logits[i]
            ex_pred_label = [id2label[p] for p in pred_labels[i]]

            # Counting consecutive similar predictions
            patience = [0] * len(ex_pred_label)
            for _ in range(len(ex_pred_label)):
                patience = [0] + [patience[i-1]+1 if ex_pred_label[i-1] == ex_pred_label[i] else 0 for i in range(1, len(ex_pred_label))]
            if len(id2label.keys()) == 1:
                # Regression
                out_dict = {
                    "ind": i,
                    "layer_logits": ex_logits.tolist(),
                    "gold_label": gold_labels[i],
                    }
            else:
                out_dict = {
                    "ind": i,
                    "layer_logits": ex_logits.tolist(),
                    "layer_probs": torch.softmax(torch.tensor(ex_logits), -1).tolist(),
                    "predicted_labels": ex_pred_label,
                    "patience": patience,
                    "gold_label_ind": gold_labels[i],
                    "gold_label": id2label[gold_labels[i]],
                }
            if meta_logits is not None:
                ex_meta_logits = meta_logits[i]
                out_dict["layer_meta_logits"] = ex_meta_logits.tolist(),
                out_dict["layer_meta_probs"] = torch.softmax(torch.tensor(ex_meta_logits),-1)[:,1].tolist(),

            writer.write(json.dumps(out_dict) + "\n")


class ECELoss(nn.Module):
    """
    From: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = torch.softmax(logits, dim=-1)
        confidences, predictions = torch.max(softmaxes, -1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
