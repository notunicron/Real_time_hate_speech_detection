import os
import json
import numpy as np
import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report, accuracy_score # f1_score, precision_score, recall_score not directly used but good to have
import logging
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_CHECKPOINT = "bert-base-uncased"
DATASET_FILE = "./dataset.json" # Path to your single JSON file
OUTPUT_DIR = "./hatexplain_word_level_model_embedded_rationales"
LABEL_LIST = ["O", "B-HATE", "I-HATE"] # Our BIO tags
LABEL_ENCODING_DICT = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
MAX_LENGTH = 128 # Max sequence length for tokenizer
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
TEST_SPLIT_RATIO = 0.1 # Use 10% of data for testing
VALID_SPLIT_RATIO = 0.1 # Use 10% of remaining for validation

# --- 1. Load and Prepare HateXplain Data (Modified) ---

def load_hatexplain_data_embedded(dataset_file):
    """Loads a single dataset JSON where rationales are embedded."""
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Ensure '{os.path.basename(dataset_file)}' exists at {dataset_file}")

    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    processed_data = []
    skipped_count = 0
    for post_id, post_data in dataset.items():
        # Ensure basic structure exists
        if 'annotators' not in post_data or 'post_tokens' not in post_data or 'rationales' not in post_data:
            logger.warning(f"Skipping post {post_id}: Missing essential keys (annotators, post_tokens, or rationales).")
            skipped_count += 1
            continue

        # Determine majority label for the post
        post_labels = [anno['label'] for anno in post_data['annotators']]
        if not post_labels:
             logger.warning(f"Skipping post {post_id}: No annotator labels found.")
             skipped_count += 1
             continue
        majority_label = Counter(post_labels).most_common(1)[0][0]

        rationale_word_indices = set() # Use a set to automatically handle duplicates

        # Only process rationales if the majority label is hate/offensive
        if majority_label in ['hatespeech', 'offensive']:
            # Check if number of rationales matches number of annotators
            if len(post_data['annotators']) != len(post_data['rationales']):
                logger.warning(f"Skipping post {post_id}: Mismatch between annotator count ({len(post_data['annotators'])}) and rationale count ({len(post_data['rationales'])}).")
                # Decide how to handle: skip, or try to align if possible (skipping is safer)
                skipped_count += 1
                continue # Skip this entry

            # Iterate through annotators and their corresponding rationale mask
            for i, annotator in enumerate(post_data['annotators']):
                annotator_label = annotator['label']
                rationale_mask = post_data['rationales'][i]

                # Check if mask length matches token length
                if len(rationale_mask) != len(post_data['post_tokens']):
                    logger.warning(f"Skipping post {post_id}, annotator {i}: Rationale mask length ({len(rationale_mask)}) mismatch with token count ({len(post_data['post_tokens'])}).")
                    # This specific rationale might be unusable, but we could potentially continue with others
                    # For safety, we might skip the whole post if any rationale is bad, or just skip this annotator's rationale
                    continue # Skip this specific annotator's rationale

                # Consider this rationale only if the annotator labeled it as hate/offensive
                if annotator_label in ['hatespeech', 'offensive']:
                    for token_idx, is_rationale in enumerate(rationale_mask):
                        if is_rationale == 1: # If the mask value is 1
                            rationale_word_indices.add(token_idx) # Add the index to our set

        processed_data.append({
            "id": post_id,
            "tokens": post_data['post_tokens'],
            "majority_label": majority_label,
            # Convert set to sorted list for consistency
            "rationale_word_indices": sorted(list(rationale_word_indices)),
        })

    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} entries due to missing keys or inconsistencies.")
    return processed_data

logger.info(f"Loading data from {DATASET_FILE}...")
raw_data = load_hatexplain_data_embedded(DATASET_FILE)
logger.info(f"Loaded {len(raw_data)} examples.")

# --- Define features (remains the same) ---
intitial_features = Features({
    'id': Value('string'),
    'tokens': Sequence(Value('string')),
    'majority_label': Value('string'),
    'rationale_word_indices': Sequence(Value('int32'))
})

# Convert to Hugging Face Dataset
logger.info("Creating initial Dataset object...")
hf_dataset = Dataset.from_list(raw_data, features=intitial_features)
logger.info("Initial Dataset object created.")

# --- 2. Tokenization and Label Alignment ---

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_and_align_labels(examples):
    """Tokenizes text and aligns rationales to BIO tags for subwords."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    labels = []
    for i, token_list in enumerate(examples["tokens"]):
        rationale_indices = set(examples["rationale_word_indices"][i])
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx in rationale_indices:
                if word_idx != previous_word_idx:
                    label_ids.append(LABEL_ENCODING_DICT["B-HATE"])
                else:
                    label_ids.append(LABEL_ENCODING_DICT["I-HATE"])
            else:
                label_ids.append(LABEL_ENCODING_DICT["O"])
            previous_word_idx = word_idx

        label_ids.extend([-100] * (MAX_LENGTH - len(label_ids)))
        labels.append(label_ids[:MAX_LENGTH])

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

logger.info("Tokenizing and aligning labels...")
# --- Columns to remove might slightly change if original data had different keys, but these seem correct ---
columns_to_remove = ['tokens', 'majority_label', 'rationale_word_indices', 'id']
tokenized_dataset = hf_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=columns_to_remove
)
logger.info("Tokenization complete.")
logger.info(f"Example processed input keys: {tokenized_dataset.column_names}")
# Check if labels exist after processing
if 'labels' in tokenized_dataset.column_names:
    logger.info(f"Example labels: {tokenized_dataset[0]['labels']}")
else:
    logger.error("Labels column not found after tokenization! Check the 'tokenize_and_align_labels' function.")


# --- 3. Split Data (remains the same) ---
if 'train' not in tokenized_dataset or 'test' not in tokenized_dataset:
     logger.info("Splitting dataset into train, validation, and test sets.")
     shuffled_dataset = tokenized_dataset.shuffle(seed=42)
     train_test_split = shuffled_dataset.train_test_split(test_size=TEST_SPLIT_RATIO)
     valid_ratio_adjusted = VALID_SPLIT_RATIO / (1 - TEST_SPLIT_RATIO)
     train_valid_split = train_test_split['train'].train_test_split(test_size=valid_ratio_adjusted)

     final_datasets = DatasetDict({
         'train': train_valid_split['train'],
         'validation': train_valid_split['test'],
         'test': train_test_split['test']
     })
     logger.info(f"Dataset splits: {final_datasets}")
else:
     logger.info("Using existing dataset splits.")
     final_datasets = tokenized_dataset

# --- 4. Model Initialization (remains the same) ---
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(LABEL_LIST),
    id2label=ID_TO_LABEL,
    label2id=LABEL_ENCODING_DICT,
    ignore_mismatched_sizes=True
)
logger.info("Model loaded.")

# --- 5. Define Metrics (remains the same) ---
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[ID_TO_LABEL[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [ID_TO_LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Use seqeval's classification_report
    report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)

    # Extract overall metrics (micro avg is suitable for token classification)
    results = {
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
        "f1": report["micro avg"]["f1-score"],
        "accuracy": accuracy_score(true_labels, true_predictions), # Overall token accuracy
    }

    # Optionally add per-class metrics if needed
    for label_name in ['HATE']: # Focus on the main entity type
         if label_name in report:
            results[f"{label_name}_precision"] = report[label_name]["precision"]
            results[f"{label_name}_recall"] = report[label_name]["recall"]
            results[f"{label_name}_f1"] = report[label_name]["f1-score"]


    return results

# --- 6. Training ---
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_dir=f'{OUTPUT_DIR}/logs', # Log within output dir
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1", # Use overall F1 for selecting best model
    push_to_hub=False,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=final_datasets["train"],
    eval_dataset=final_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

logger.info("Starting training...")
train_result = trainer.train()
logger.info("Training finished.")

# --- 7. Evaluation on Test Set ---
logger.info("Evaluating on the test set...")
test_results = trainer.evaluate(eval_dataset=final_datasets["test"])
logger.info(f"Test Set Evaluation Results: {test_results}")

# --- 8. Save the final model and tokenizer ---
logger.info(f"Saving the best model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

output_test_results_file = os.path.join(OUTPUT_DIR, "test_results.txt")
with open(output_test_results_file, "w") as writer:
    logger.info("***** Test results *****")
    for key, value in sorted(test_results.items()):
        logger.info(f"  {key} = {value}")
        writer.write(f"{key} = {value}\n")

logger.info("Script finished successfully.")

# --- How to use the trained model (example) ---
# from transformers import pipeline

# word_hate_pipe = pipeline(
#     "token-classification",
#     model=OUTPUT_DIR,
#     tokenizer=OUTPUT_DIR,
#     aggregation_strategy="simple" # Groups subwords nicely
# )

# text1 = "You are such an idiot, go back where you came from!"
# text2 = "This is a normal sentence about cats."

# results1 = word_hate_pipe(text1)
# results2 = word_hate_pipe(text2)

# print(f"Results for: '{text1}'")
# print(results1)

# print(f"\nResults for: '{text2}'")
# print(results2)