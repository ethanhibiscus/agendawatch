import os
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from functools import partial
from tqdm import tqdm
import pickle
from models.SDR.SDR import SDR
from data.datasets import CustomTextDatasetParagraphsSentencesTest
from models.reco.recos_utils import sim_matrix
from models.reco.hierarchical_reco import mean_non_pad_value
from data.data_utils import reco_sentence_test_collate

# Define the hparams as per your configuration
class HParams:
    dataset_name = 'custom_dataset'
    limit_tokens = 512
    num_data_workers = 4
    test_batch_size = 1

hparams = HParams()

def main():
    print("Starting inference process...")

    # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model_dir = "/home/ethanhsu/03_07_2024-23_10_34"
    checkpoint_file = os.path.join(model_dir, "epoch=10.ckpt")

    print(f"Loading model from checkpoint: {checkpoint_file}")
    model = SDR.load_from_checkpoint(checkpoint_path=checkpoint_file, hparams=hparams)

    # Prepare the dataset
    print("Preparing dataset...")
    test_dataset = CustomTextDatasetParagraphsSentencesTest(tokenizer=tokenizer, hparams=hparams, dataset_name=hparams.dataset_name, block_size=hparams.limit_tokens, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=hparams.test_batch_size, collate_fn=partial(reco_sentence_test_collate, tokenizer=tokenizer), num_workers=hparams.num_data_workers)

    # Create output directory if it doesn't exist
    output_dir = "./inference_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Extract features for all documents
    print("Extracting features for all documents...")
    all_features = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting Features"):
            section_out = []
            for section in batch[0]:
                sentences = []
                for sentence in section[0]:
                    sentence_embed = model(sentence.unsqueeze(0), masked_lm_labels=None, run_similarity=False)[5].squeeze(0)
                    sentences.append(sentence_embed.mean(0))
                section_out.append(torch.stack(sentences))
            all_features.append(section_out)

    # Save extracted features
    features_file = os.path.join(output_dir, "all_features.pkl")
    print(f"Saving extracted features to {features_file}")
    with open(features_file, "wb") as f:
        pickle.dump(all_features, f)

    # Compute similarity between the source document and all other documents
    print("Computing similarity scores...")
    source_doc_features = all_features[0]  # Assuming the first document is the source document

    similarity_scores = []
    for i, candidate_features in tqdm(enumerate(all_features), desc="Computing Similarity Scores", total=len(all_features)):
        sim = sim_matrix(source_doc_features, candidate_features)
        sim[sim.isnan()] = float("-Inf")
        score = mean_non_pad_value(sim, axis=-1, pad_value=float("-Inf")).max(-1)[0].mean().item()
        similarity_scores.append((i, score))

    # Rank the documents based on similarity scores
    print("Ranking documents based on similarity scores...")
    ranked_documents = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Save similarity scores
    scores_file = os.path.join(output_dir, "similarity_scores.pkl")
    print(f"Saving similarity scores to {scores_file}")
    with open(scores_file, "wb") as f:
        pickle.dump(similarity_scores, f)

    # Output the ranked documents
    print("Final ranked documents:")
    for doc_id, score in ranked_documents:
        print(f"Document {doc_id} - Similarity Score: {score}")

    # Save ranked documents
    ranked_file = os.path.join(output_dir, "ranked_documents.pkl")
    print(f"Saving ranked documents to {ranked_file}")
    with open(ranked_file, "wb") as f:
        pickle.dump(ranked_documents, f)

    print("Inference process completed successfully.")

if __name__ == "__main__":
    main()
