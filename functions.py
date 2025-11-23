from transformers import pipeline
from tqdm import tqdm
import evaluate
from typing import Dict, List, Any, Union


def create_prompt(sample: Dict[str, str], summary: bool = False) -> str:
    """Constructs a prompt string for training or inference.

    For training, two variants are used:
    - ``summary=False``: full instruction prompt ending with "Summary:\n"
    - ``summary=True``: only the ground-truth summary text (for label generation)

    For inference, only ``summary=False`` is used to generate the input prompt.

    Args:
        sample (Dict[str, str]): A dataset sample containing at least "article" 
            and/or "abstract" keys.
        summary (bool, optional): If True, returns only the abstract text.
            If False (default), returns the full system prompt + article + "Summary:\n".

    Returns:
        str: Formatted prompt string.

    Examples:
        >>> sample = {"article": "This is an article.", "abstract": "Short summary."}
        >>> create_prompt(sample, summary=False)
        'You are a helpful assistant...\\nArticle:\\nThis is an article.\\nSummary:\\n'
        >>> create_prompt(sample, summary=True)
        'Short summary.'
    """
    if summary:
        prompt = sample["abstract"]
    else:
        system_prompt = "You are a helpful assistant who writes concise, factual summaries of articles. Summarize the following article into a few sentences."
        prompt = f"{system_prompt}\nArticle:\n{sample['article']}\nSummary:\n"
    return prompt


def compute_rouge(predictions: List[str], samples: List[Dict[str, str]]) -> Dict[str, float]:
    """Computes ROUGE metrics (1, 2, L, Lsum) between predictions and references.

    Uses the Hugging Face `evaluate` library's "rouge" implementation.
    ROUGE-Lsum is particularly relevant for multi-sentence summarization.

    Args:
        predictions (List[str]): List of generated summaries.
        samples (List[Dict[str, str]]): List of dataset samples, each must contain 
            an "abstract" key with the reference summary.

    Returns:
        Dict[str, float]: Dictionary with keys "rouge1", "rouge2", "rougeL", "rougeLsum"
        and corresponding F1 scores in range [0.0, 1.0].

    Note:
        This function expects samples to be in the same order as predictions.
    """
    rouge = evaluate.load("rouge")
    references = [s["abstract"] for s in samples]
    return rouge.compute(predictions=predictions, references=references)


def evaluate_test(data: List[Dict[str, str]], model: Any, tokenizer: Any) -> Dict[str, float]:
    """Evaluates a model on a test dataset using ROUGE metrics.

    Performs batched inference with greedy decoding, then computes ROUGE scores.
    Designed to avoid OOM by using moderate batch sizes and short generation.

    Args:
        data (List[Dict[str, str]]): Test dataset (list of samples with "article"/"abstract").
        model (Any): A Hugging Face-compatible causal language model (e.g., PeftModel).
        tokenizer (Any): Corresponding tokenizer.

    Returns:
        Dict[str, float]: ROUGE scores (same format as `compute_rouge`).

    Note:
        Uses `do_sample=False` (greedy decoding) for reproducibility.
        Max generation length is capped at 256 tokens.
    """
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, do_sample=False)
    # Create inference prompts for all samples in "data"
    prompts = [create_prompt(sample, summary=False) for sample in data]

    batch_size = 32
    preds = []
    # Run inference for all prompts
    for idx in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[idx : idx + batch_size]
        outs = pipe(batch, max_new_tokens=256, return_full_text=False)
        # Collect generated summaries
        preds.extend(item[0]["generated_text"].strip() for item in outs)
    
    # Compute ROUGE score for all predictions
    return compute_rouge(preds, data)


def prepare_data(sample: Dict[str, str], tokenizer: Any) -> Dict[str, List[int]]:
    """Preprocesses a single dataset sample for causal language modeling.

    Converts raw article/abstract into tokenized input_ids, attention_mask, and labels.
    Implements instruction-tuning format with loss masking on the prompt part.

    Steps:
        1. Safely extract and clean article/abstract text
        2. Format prompt and summary strings
        3. Tokenize separately (with/without special tokens)
        4. Concatenate and mask labels (-100 for prompt tokens)
        5. Truncate to max length (1024) and ensure at least one learnable token

    Args:
        sample (Dict[str, str]): Dataset sample with "article" and "abstract" keys.
        tokenizer (Any): Hugging Face tokenizer (must support `__call__`).

    Returns:
        Dict[str, List[int]]: Dictionary with:
            - "input_ids": token IDs of full sequence (prompt + summary)
            - "labels": same as input_ids, but prompt tokens replaced with -100
            - "attention_mask": 1 for real tokens, 0 for padding

    Raises:
        AssertionError: If final sequences have mismatched lengths.

    Note:
        Designed for use with `Dataset.map(..., batched=False)`.
    """
    article = str(sample["article"]).strip() or "No article."
    abstract = str(sample["abstract"]).strip() or "No summary available."
    
    prompt = create_prompt({"article": article}, False)  # system + article + "Summary:\n"
    summary_text = create_prompt({"abstract": abstract}, True)  # Summary text
    
    prompt_tokens = tokenizer(
        prompt, truncation=True, max_length=768, add_special_tokens=True
    )
    summary_tokens = tokenizer(
        summary_text, truncation=True, max_length=256, add_special_tokens=False
    )
    
    # Ensure summary is not empty (critical for non-NaN loss)
    if len(summary_tokens["input_ids"]) == 0:
        summary_tokens = tokenizer(".", add_special_tokens=False)
    
    input_ids = prompt_tokens["input_ids"] + summary_tokens["input_ids"]
    # Mask user input and leave only assistant response to compute loss
    labels = [-100] * len(prompt_tokens["input_ids"]) + summary_tokens["input_ids"]
    attention_mask = prompt_tokens["attention_mask"] + summary_tokens["attention_mask"]
    
    # Truncate if needed
    MAX_LEN = 1024
    input_ids = input_ids[:MAX_LEN]
    labels = labels[:MAX_LEN]
    attention_mask = attention_mask[:MAX_LEN]
    
    # Prevent completely masked labels (would cause NaN loss)
    if all(l == -100 for l in labels):
        labels[-1] = tokenizer.eos_token_id
    
    # Ensure type consistency for Hugging Face collators
    result = {
        "input_ids": [int(x) for x in input_ids],
        "labels": [int(x) for x in labels],
        "attention_mask": [int(x) for x in attention_mask],
    }
    
    # Safety check
    assert len(result["input_ids"]) == len(result["labels"]) == len(result["attention_mask"]), \
        "Mismatched sequence lengths in prepared sample"
    
    return result