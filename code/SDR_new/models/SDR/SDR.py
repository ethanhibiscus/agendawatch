from data.datasets import CustomTextDataset
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer, RobertaForMaskedLM
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from utils.argparse_init import str2bool
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F


class SDR(LightningModule):
    def __init__(self, hparams):
        super(SDR, self).__init__()
        self.save_hyperparameters(hparams)

        self.config = RobertaConfig.from_pretrained(hparams.config_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(hparams.tokenizer_name)
        self.roberta = RobertaModel.from_pretrained(hparams.model_name_or_path, config=self.config)

        self.projection_head = nn.Linear(self.config.hidden_size, hparams.projection_dim)
        self.lm_head = RobertaForMaskedLM.from_pretrained(hparams.model_name_or_path).lm_head

        self.hparams = hparams

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]
        embeddings = self.projection_head(pooled_output)
        return embeddings

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        embeddings = self(inputs)

        # MLM Loss
        mlm_loss = self.compute_mlm_loss(inputs)

        # Contrastive Loss
        contrastive_loss = self.compute_contrastive_loss(embeddings, labels)

        # Total Loss
        total_loss = mlm_loss + self.hparams.contrastive_weight * contrastive_loss

        self.log('train_loss', total_loss)
        return total_loss

    def compute_mlm_loss(self, inputs):
        masked_inputs, labels = self.mask_tokens(inputs)
        outputs = self.roberta(masked_inputs)
        prediction_scores = self.lm_head(outputs[0])
        mlm_loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        return mlm_loss

    def compute_contrastive_loss(self, embeddings, labels):
        cosine_sim = nn.CosineSimilarity(dim=-1)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        similarity_matrix = cosine_sim(embeddings.unsqueeze(1), embeddings.unsqueeze(0))
        positive_pairs = similarity_matrix * mask
        negative_pairs = similarity_matrix * (1 - mask)
        contrastive_loss = -torch.log(positive_pairs + 1e-6).mean() + torch.log(negative_pairs + 1e-6).mean()
        return contrastive_loss

    def mask_tokens(self, inputs):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.hparams.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.total_steps
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = CustomTextDataset(directory=self.hparams.data_dir, tokenizer=self.tokenizer, block_size=self.hparams.block_size)
        return DataLoader(dataset, batch_size=self.hparams.train_batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        dataset = CustomTextDataset(directory=self.hparams.data_dir, tokenizer=self.tokenizer, block_size=self.hparams.block_size)
        return DataLoader(dataset, batch_size=self.hparams.val_batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        dataset = CustomTextDataset(directory=self.hparams.data_dir, tokenizer=self.tokenizer, block_size=self.hparams.block_size)
        return DataLoader(dataset, batch_size=self.hparams.test_batch_size, shuffle=False, num_workers=4)

    def compute_similarity(self, source_embedding, target_embeddings):
        cosine_sim = nn.CosineSimilarity(dim=-1)
        similarity_scores = cosine_sim(source_embedding.unsqueeze(0), target_embeddings)
        return similarity_scores

    def rank_documents(self, source_document, all_documents):
        source_embedding = self(source_document)
        target_embeddings = torch.stack([self(doc) for doc in all_documents])
        similarity_scores = self.compute_similarity(source_embedding, target_embeddings)
        ranked_indices = similarity_scores.argsort(descending=True)
        return ranked_indices

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--data_dir", type=str, help="Path to the text files directory")
        parser.add_argument("--block_size", type=int, default=512, help="Maximum input sequence length")
        parser.add_argument("--projection_dim", type=int, default=128, help="Dimension of the projection head")
        parser.add_argument("--mlm_probability", type=float, default=0.15, help="Probability for masking tokens")
        parser.add_argument("--contrastive_weight", type=float, default=0.1, help="Weight for the contrastive loss")
        parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
        parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
        parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps for learning rate scheduler")
        parser.add_argument("--total_steps", type=int, default=10000, help="Total training steps")
        parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
        parser.add_argument("--val_batch_size", type=int, default=16, help="Validation batch size")
        parser.add_argument("--test_batch_size", type=int, default=16, help="Test batch size")
        return parser
