import os
from pathlib import Path
from datasets import load_dataset
from glob import glob
import json

# Step 1: Load dataset in streaming mode
DATASET_FILE_PATHS = list(glob(str(Path(os.path.abspath(__file__)).parent / "dataset/bert_pretraining_data/*.jsonl")))

dataset = load_dataset(
    "json",
    data_files=DATASET_FILE_PATHS,
    streaming=True  # Prevents memory overflow
)

dataset = dataset["train"]  # Get the train split

# Step 2: Process dataset in smaller chunks
batch_size = 1_000_000  # Adjust based on memory limits
buffer = []
chunks = []
chunk_count = 0

output_dir = Path("processed_chunks")
output_dir.mkdir(exist_ok=True)

# Save dataset in smaller JSONL files
for i, example in enumerate(dataset):
    buffer.append(example)

    if len(buffer) >= batch_size:
        chunk_path = output_dir / f"chunk_{chunk_count}.jsonl"
        with open(chunk_path, "w") as f:
            for record in buffer:
                f.write(json.dumps(record) + "\n")

        print(f"Saved {chunk_path}")
        chunks.append(str(chunk_path))  # Keep track of saved files
        buffer = []
        chunk_count += 1

# Save remaining records
if buffer:
    chunk_path = output_dir / f"chunk_{chunk_count}.jsonl"
    with open(chunk_path, "w") as f:
        for record in buffer:
            f.write(json.dumps(record) + "\n")
    print(f"Saved {chunk_path}")
    chunks.append(str(chunk_path))

# Step 3: Reload and push to Hugging Face Hub
dataset = load_dataset("json", data_files={"train": chunks})  # Reload dataset from saved chunks
dataset.push_to_hub("shahrukhx01/wikipedia-bookscorpus-en-preprocessed")  # Push to HF Hub
