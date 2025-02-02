# BERT Pretraining Data Preprocessing Pipeline

This repository contains a pipeline for preprocessing Wikipedia and BookCorpus datasets for BERT pretraining. The pipeline consists of two main stages: preprocessing and normalization.

## Pipeline Overview

### Stage 1: Preprocessing
- Language filtering (English only)
- Text chunking (820 characters per chunk, preserving sentence boundaries)
- Outputs JSONL files with preprocessed chunks

```bash
python preprocess.py
```

### Stage 2: Normalization
- Text normalization (lowercase, remove accents and non-English characters)
- Length filtering (removes texts < 200 characters)
- Outputs normalized JSONL files

```bash
python normalize.py
```

### Stage 3: Publishing (Optional)
- Creates shuffled chunks of 1M samples each
- Publishes to Hugging Face Hub

```bash
python publish_dataset_to_hf_hub.py
```

## Configuration

Create YAML config files in the `configs` directory path seggregated by normalization and data preparation (preprocessing).

## Features

- Distributed processing using Ray actors
- Batch processing to handle large datasets
- Sentence boundary preservation during chunking
- Configurable language filtering and chunk sizes
- Progress logging with loguru
- Type checking with pydantic

## Output Format

Each processed sample is stored in JSONL format:
```json
{"text": "processed text chunk", "is_filtered_out": false}
```

The final dataset can be optionally published to the Hugging Face Hub for easy access with the name: [wikipedia-bookscorpus-en-preprocessed](https://huggingface.co/datasets/shahrukhx01/wikipedia-bookscorpus-en-preprocessed)
