from argparse import ArgumentParser
import logging
import math
from pytorch_metric_learning.distances.lp_distance import LpDistance

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import gelu
from pytorch_metric_learning import miners, losses, reducers

from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_bert import BertLayerNorm, BertPreTrainedModel
from transformers.modeling_roberta import RobertaModel, RobertaLMHead
from pytorch_metric_learning.distances import CosineSimilarity


class SimilarityModeling(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, hparams):
        """
        Initialize the SimilarityModeling model.

        Args:
            config (RobertaConfig): Configuration for the RoBERTa model.
            hparams (Namespace): Hyperparameters for the model, including various settings for training and evaluation.
        """
        super().__init__(config)
        self.hparams = hparams
        config.output_hidden_states = True

        # Initialize the RoBERTa model and the LM head
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # Metric selection for similarity calculation
        if self.hparams.metric_for_similarity == "cosine":
            self.metric = CosineSimilarity()
            pos_margin, neg_margin = 1, 0
            neg_margin = -1 if (getattr(self.hparams, "metric_loss_func", "ContrastiveLoss") == "CosineLoss") else 0
        elif self.hparams.metric_for_similarity == "norm_euc":
            self.metric = LpDistance(normalize_embeddings=True, p=2)
            pos_margin, neg_margin = 0, 1

        # Reducer configuration
        self.reducer = reducers.DoNothingReducer()
        
        # Miner function configuration for hard positive/negative mining
        if self.hparams.hard_mine:
            self.miner_func = miners.MultiSimilarityMiner()
        else:
            self.miner_func = miners.BatchEasyHardMiner(
                pos_strategy=miners.BatchEasyHardMiner.ALL,
                neg_strategy=miners.BatchEasyHardMiner.ALL,
                distance=CosineSimilarity(),
            )

        # Similarity loss function configuration
        if getattr(self.hparams, "metric_loss_func", "ContrastiveLoss") in ["ContrastiveLoss", "CosineLoss"]:
            self.similarity_loss_func = losses.ContrastiveLoss(
                pos_margin=pos_margin, neg_margin=neg_margin, distance=self.metric
            )
        else:
            self.similarity_loss_func = losses.TripletMarginLoss(margin=1, distance=self.metric)

    def get_output_embeddings(self):
        """
        Returns the output embeddings from the LM head.

        Returns:
            nn.Module: The LM head decoder.
        """
        return self.lm_head.decoder

    @staticmethod
    def mean_mask(features, mask):
        """
        Calculate the mean of the features, considering only non-masked values.

        Args:
            features (torch.Tensor): The input feature tensor.
            mask (torch.Tensor): The mask tensor.

        Returns:
            torch.Tensor: The mean of the non-masked features.
        """
        return (features * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)

    def forward(
        self,
        input_ids=None,
        sample_labels=None,
        samples_idxs=None,
        track_sim_dict=None,
        non_masked_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        labels=None,
        output_hidden_states=False,
        return_dict=False,
        run_similarity=False,
        run_mlm=True,
    ):
        """
        Forward pass of the model. Supports both masked language modeling (MLM) and similarity learning.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            sample_labels (torch.Tensor): Labels for the samples, used in similarity learning.
            samples_idxs (torch.Tensor): Indices of the samples.
            track_sim_dict (dict): Dictionary for tracking similarities.
            non_masked_input_ids (torch.Tensor): Non-masked input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor): Token type IDs.
            position_ids (torch.Tensor): Position IDs.
            head_mask (torch.Tensor): Head mask.
            inputs_embeds (torch.Tensor): Input embeddings.
            masked_lm_labels (torch.Tensor): Labels for masked language modeling.
            labels (torch.Tensor): General labels.
            output_hidden_states (bool): Whether to output hidden states.
            return_dict (bool): Whether to return a dictionary.
            run_similarity (bool): Whether to run the similarity calculation.
            run_mlm (bool): Whether to run masked language modeling.

        Returns:
            tuple: Contains the MLM loss, similarity loss, and outputs from the RoBERTa model.
        """
        # Masked Language Modeling (MLM)
        if run_mlm:
            outputs = list(
                self.roberta(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            )
            sequence_output = outputs[0]
            prediction_scores = self.lm_head(sequence_output)
            outputs = (prediction_scores, None, sequence_output)

            masked_lm_loss = torch.zeros(1, device=prediction_scores.device).float()

            if (masked_lm_labels is not None and (not (masked_lm_labels == -100).all())) and self.hparams.mlm:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            else:
                masked_lm_loss = 0
        else:
            # If not running MLM, return zeros for MLM outputs
            outputs = (
                torch.zeros([*input_ids.shape, 50265]).to(input_ids.device).float(),
                None,
                torch.zeros([*input_ids.shape, 1024]).to(input_ids.device).float(),
            )
            masked_lm_loss = torch.zeros(1)[0].to(input_ids.device).float()

        # Similarity Loss Calculation
        if run_similarity:
            non_masked_outputs = self.roberta(
                non_masked_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            non_masked_seq_out = non_masked_outputs[0]

            meaned_sentences = non_masked_seq_out.mean(1)
            miner_output = list(self.miner_func(meaned_sentences, sample_labels))

            sim_loss = self.similarity_loss_func(meaned_sentences, sample_labels, miner_output)
            outputs = (masked_lm_loss, sim_loss, torch.zeros(1)) + outputs
        else:
            outputs = (
                masked_lm_loss,
                torch.zeros(1)[0].to(input_ids.device).float(),
                torch.zeros(1)[0].to(input_ids.device).float(),
            ) + outputs

        return outputs
