import os
import random
import torch
import pickle
from transformers import RobertaTokenizer
from models.SDR.SDR import SDR
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from utils.torch_utils import to_numpy
from utils.argparse_init import init_parse_argparse_default_params
from data.datasets import CustomTextDatasetParagraphsSentencesTest
from models.reco.hierarchical_reco import vectorize_reco_hierarchical
from pytorch_lightning.utilities.seed import seed_everything

def load_model_and_tokenizer(checkpoint_dir):
    print("Loading model and tokenizer...")
    checkpoint_path = os.path.join(checkpoint_dir, 'last.ckpt')
    checkpoint = torch.load(checkpoint_path)
    hparams = checkpoint['hyper_parameters']
    model = SDR.load_from_checkpoint(checkpoint_path, hparams=hparams)
    tokenizer = RobertaTokenizer.from_pretrained(hparams['config_name'])
    print("Model and tokenizer loaded.")
    return model, tokenizer

def load_dataset(tokenizer, hparams):
    print("Loading dataset...")
    dataset = CustomTextDatasetParagraphsSentencesTest(
        tokenizer=tokenizer,
        hparams=hparams,
        dataset_name=hparams['dataset_name'],
        block_size=hparams['block_size'],
        mode="test",
    )
    print("Dataset loaded.")
    return dataset

def compute_similarity_scores(model, tokenizer, dataset, hparams):
    print("Computing similarity scores...")
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=hparams['test_batch_size'],
        num_workers=hparams['num_data_workers'],
        collate_fn=partial(reco_sentence_test_collate, tokenizer=tokenizer),
    )

    outputs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            section_out = []
            for section in tqdm(batch[0], desc="Processing sections", leave=False):
                sentences = []
                for sentence in section[0][:8]:
                    sentences.append(sentence.mean(0))
                section_out.append(torch.stack(sentences))
            outputs.append((section_out, batch[0][0][1][0]))
    
    print("Similarity scores computed.")
    return outputs

def rank_documents(model, outputs):
    print("Ranking documents...")
    titles = [output[1] for output in outputs]
    section_sentences_features = [output[0] for output in outputs]
    recos, metrics = vectorize_reco_hierarchical(
        all_features=section_sentences_features,
        titles=titles,
        gt_path="",
        output_path=""
    )
    print("Documents ranked.")
    return recos

def save_results(recos, output_dir):
    print("Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ranked_documents.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(recos, f)
    print(f"Results saved to {output_path}")

def main():
    checkpoint_dir = os.path.expanduser('~/03_07_2024-23_10_34')
    output_dir = './inference_outputs'

    print("Initializing...")
    seed_everything(42)
    
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir)
    hparams = model.hparams
    dataset = load_dataset(tokenizer, hparams)
    
    outputs = compute_similarity_scores(model, tokenizer, dataset, hparams)
    recos = rank_documents(model, outputs)
    
    save_results(recos, output_dir)
    print("Inference completed.")

if __name__ == "__main__":
    main()
