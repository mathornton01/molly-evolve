"""
Domain data loading for continual learning experiments.

Provides predefined domain datasets and custom data loading.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("molly_evolution")

# Predefined domain datasets (HuggingFace)
DOMAIN_CONFIGS = {
    "general": {
        "dataset": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "text_field": "text",
        "max_samples": 5000,
        "description": "General English text (WikiText-103)",
    },
    "code": {
        "dataset": "sahil2801/CodeAlpaca-20k",
        "config": None,
        "split": "train",
        "text_field": "output",
        "max_samples": 5000,
        "description": "Source code (CodeAlpaca)",
    },
    "legal": {
        "dataset": "joelito/eurlex",
        "config": None,
        "split": "train",
        "text_field": "text",
        "max_samples": 500,
        "description": "EU legal text (EurLex)",
    },
    "medical": {
        "dataset": "medalpaca/medical_meadow_medqa",
        "config": None,
        "split": "train",
        "text_field": "input",
        "max_samples": 2000,
        "description": "Medical Q&A",
    },
    "science": {
        "dataset": "scientific_papers",
        "config": "arxiv",
        "split": "train",
        "text_field": "article",
        "max_samples": 500,
        "description": "Scientific papers (arXiv)",
    },
    "finance": {
        "dataset": "financial_phrasebank",
        "config": "sentences_allagree",
        "split": "train",
        "text_field": "sentence",
        "max_samples": 500,
        "description": "Financial sentiment text",
    },
}

# Built-in quick-test data (no download needed)
QUICKTEST_DATA = {
    "general": (
        "The history of artificial intelligence began in antiquity, with myths and "
        "stories of artificial beings endowed with intelligence. The seeds of modern "
        "AI were planted by philosophers who attempted to describe the process of human "
        "thinking as the mechanical manipulation of symbols. This work culminated in "
        "the invention of the programmable digital computer in the 1940s. "
    ) * 5,
    "code": (
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) "
        "+ fibonacci(n-2)\n\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
        "    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n"
        "    return quicksort(left) + middle + quicksort(right)\n\n"
        "class BinarySearchTree:\n    def __init__(self, value):\n        self.value = value\n"
        "        self.left = None\n        self.right = None\n"
    ) * 5,
    "legal": (
        "The defendant is hereby charged with violation of Section 1983 of Title 42 "
        "of the United States Code. The plaintiff alleges that the defendant, acting "
        "under color of state law, deprived the plaintiff of rights secured by the "
        "Constitution. The court must determine whether qualified immunity shields "
        "the defendant from liability. Under the doctrine of stare decisis, the court "
        "is bound by the precedent established in prior rulings. "
    ) * 5,
    "medical": (
        "The patient presents with acute myocardial infarction characterized by "
        "ST-segment elevation in leads V1-V4. Troponin I levels are elevated at "
        "15.2 ng/mL. The recommended treatment includes immediate percutaneous "
        "coronary intervention with dual antiplatelet therapy consisting of aspirin "
        "and a P2Y12 inhibitor. Echocardiography reveals reduced left ventricular "
        "ejection fraction of 35%. "
    ) * 5,
}


def load_domain_data(domain: str, tokenizer, max_length: int = 256,
                     n_train: int = 100, n_eval: int = 50,
                     quicktest: bool = False) -> Tuple[dict, dict]:
    """
    Load train and eval data for a domain.

    Args:
        domain: Domain name (from DOMAIN_CONFIGS or QUICKTEST_DATA).
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        n_train: Number of training examples.
        n_eval: Number of eval examples.
        quicktest: Use built-in data instead of downloading.

    Returns:
        (train_encodings, eval_encodings)
    """
    if quicktest or domain not in DOMAIN_CONFIGS:
        return _load_quicktest(domain, tokenizer, max_length, n_train, n_eval)

    return _load_hf_domain(domain, tokenizer, max_length, n_train, n_eval)


def _load_quicktest(domain, tokenizer, max_length, n_train, n_eval):
    """Load from built-in text data."""
    text = QUICKTEST_DATA.get(domain, QUICKTEST_DATA["general"])
    logger.info(f"  Loading '{domain}' (quicktest, built-in data)")

    # Replicate text to get enough tokens
    while len(text) < max_length * (n_train + n_eval):
        text = text + " " + text

    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_length * (n_train + n_eval),
                    padding=False)

    # Split into chunks
    all_ids = enc["input_ids"][0]
    chunks = []
    for i in range(0, len(all_ids) - max_length, max_length):
        chunks.append(all_ids[i:i+max_length])

    n_total = min(len(chunks), n_train + n_eval)
    n_train = min(n_train, n_total - 1)
    n_eval = min(n_eval, n_total - n_train)

    import torch
    train_ids = torch.stack(chunks[:n_train])
    eval_ids = torch.stack(chunks[n_train:n_train+n_eval])

    train_enc = {
        "input_ids": train_ids,
        "attention_mask": torch.ones_like(train_ids),
    }
    eval_enc = {
        "input_ids": eval_ids,
        "attention_mask": torch.ones_like(eval_ids),
    }

    logger.info(f"    train: {train_ids.shape[0]} x {train_ids.shape[1]} tokens")
    logger.info(f"    eval:  {eval_ids.shape[0]} x {eval_ids.shape[1]} tokens")
    return train_enc, eval_enc


def _load_hf_domain(domain, tokenizer, max_length, n_train, n_eval):
    """Load from HuggingFace datasets."""
    from datasets import load_dataset

    cfg = DOMAIN_CONFIGS[domain]
    logger.info(f"  Loading '{domain}' from {cfg['dataset']}...")

    ds_args = [cfg["dataset"]]
    if cfg.get("config"):
        ds_args.append(cfg["config"])

    max_samples = cfg.get("max_samples", n_train + n_eval + 50)
    try:
        ds = load_dataset(*ds_args, split=f"{cfg['split']}[:{max_samples}]")
    except Exception as e:
        logger.warning(f"  Failed to load '{domain}': {e}")
        logger.warning(f"  Falling back to quicktest data")
        return _load_quicktest(domain, tokenizer, max_length, n_train, n_eval)

    # Extract text
    text_field = cfg["text_field"]
    texts = [row[text_field] for row in ds if row.get(text_field)]
    texts = [t for t in texts if len(t) > 50]  # filter too-short

    if not texts:
        logger.warning(f"  No valid texts for '{domain}', falling back to quicktest")
        return _load_quicktest(domain, tokenizer, max_length, n_train, n_eval)

    # Tokenize and chunk
    import torch
    all_text = " ".join(texts)
    enc = tokenizer(all_text, return_tensors="pt", truncation=True,
                    max_length=max_length * (n_train + n_eval + 10),
                    padding=False)

    all_ids = enc["input_ids"][0]
    chunks = []
    for i in range(0, len(all_ids) - max_length, max_length):
        chunks.append(all_ids[i:i+max_length])

    n_total = min(len(chunks), n_train + n_eval)
    n_train = min(n_train, n_total - 1)
    n_eval = min(n_eval, n_total - n_train)

    train_ids = torch.stack(chunks[:n_train])
    eval_ids = torch.stack(chunks[n_train:n_train+n_eval])

    train_enc = {
        "input_ids": train_ids,
        "attention_mask": torch.ones_like(train_ids),
    }
    eval_enc = {
        "input_ids": eval_ids,
        "attention_mask": torch.ones_like(eval_ids),
    }

    logger.info(f"    train: {train_ids.shape[0]} x {train_ids.shape[1]} tokens")
    logger.info(f"    eval:  {eval_ids.shape[0]} x {eval_ids.shape[1]} tokens")
    return train_enc, eval_enc
