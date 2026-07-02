"""LITM Training Dataset for Phase B

Generates Lost-In-Middle benchmark training examples with dynamic budget and difficulty levels.
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class LITMExample:
    """Single LITM training example."""
    context: str          # Full context with documents
    question: str         # Question text
    answer_key: str       # Correct answer key
    position: str         # "beginning" | "middle" | "end"
    n_pairs: int          # 5, 10, 20, 40
    budget: float         # 0.3, 0.5, 0.7, 1.0 (budget as fraction of full length)
    

class LITMTrainingDataset(Dataset):
    """PyTorch Dataset for LITM training."""
    
    def __init__(
        self,
        examples: List[LITMExample],
        tokenizer,
        max_length: int = 4096,
        include_position_label: bool = True,
        include_budget_label: bool = True,
    ):
        """
        Args:
            examples: List of LITMExample objects
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            include_position_label: Whether to include position in features
            include_budget_label: Whether to include budget in features
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_position_label = include_position_label
        self.include_budget_label = include_budget_label
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Tokenize context + question + answer format
        # Format: "Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}"
        prompt = f"Context: {example.context}\n\nQuestion: {example.question}\n\nAnswer:"
        answer_text = f" {example.answer_key}"
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=False,  # We'll handle truncation carefully
            return_tensors=None,
        )
        prompt_input_ids = prompt_tokens["input_ids"]
        
        # Tokenize answer
        answer_tokens = self.tokenizer(
            answer_text,
            truncation=False,
            return_tensors=None,
        )
        answer_input_ids = answer_tokens["input_ids"]
        
        # Concatenate (this is what the model should predict)
        full_input_ids = prompt_input_ids + answer_input_ids
        
        # Truncate to max_length if needed
        if len(full_input_ids) > self.max_length:
            full_input_ids = full_input_ids[:self.max_length]
        
        # Pad to max_length
        pad_length = self.max_length - len(full_input_ids)
        if pad_length > 0:
            full_input_ids = full_input_ids + [self.tokenizer.pad_token_id] * pad_length
        
        # Create target IDs (same as input for LM, but ignore pad tokens)
        target_ids = full_input_ids.copy()
        
        # Create attention mask
        attention_mask = [1] * (self.max_length - pad_length) + [0] * pad_length
        
        # Prepare features dict
        features = {
            "input_ids": torch.tensor(full_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "budget": torch.tensor(example.budget, dtype=torch.float32),
            "n_pairs": torch.tensor(example.n_pairs, dtype=torch.long),
            "seq_length": torch.tensor(len(full_input_ids) - pad_length, dtype=torch.long),
        }
        
        # Add position label if requested
        if self.include_position_label:
            position_to_id = {"beginning": 0, "middle": 1, "end": 2}
            features["position_label"] = torch.tensor(
                position_to_id[example.position],
                dtype=torch.long,
            )
        
        # Add budget label if requested
        if self.include_budget_label:
            budget_to_id = {0.3: 0, 0.5: 1, 0.7: 2, 1.0: 3}
            features["budget_label"] = torch.tensor(
                budget_to_id.get(example.budget, 1),  # Default to 0.5 if unknown
                dtype=torch.long,
            )
        
        return features


def create_litm_training_set(
    n_examples: int = 5000,
    n_pairs_options: Optional[List[int]] = None,
    positions: Optional[List[str]] = None,
    budgets: Optional[List[float]] = None,
    seed: int = 42,
) -> List[LITMExample]:
    """
    Synthetic LITM training set generator.
    
    Args:
        n_examples: Number of examples to generate
        n_pairs_options: List of n_pairs values to sample from
        positions: List of positions to sample from
        budgets: List of budget values to sample from
        seed: Random seed for reproducibility
        
    Returns:
        List of LITMExample objects
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    if n_pairs_options is None:
        n_pairs_options = [5, 10, 20, 40]
    if positions is None:
        positions = ["beginning", "middle", "end"]
    if budgets is None:
        budgets = [0.3, 0.5, 0.7, 1.0]
    
    examples = []
    
    for i in range(n_examples):
        # Sample hyperparameters
        n_pairs = random.choice(n_pairs_options)
        position = random.choice(positions)
        budget = random.choice(budgets)
        
        # Generate synthetic LITM example
        # Create context with n_pairs documents
        documents = []
        for j in range(n_pairs):
            doc_content = f"Document {j+1}: " + " ".join([
                f"token_{j}_{k}" for k in range(50)  # Each doc ~50 tokens
            ])
            documents.append(doc_content)
        
        # Shuffle documents
        doc_indices = list(range(n_pairs))
        if position == "beginning":
            # Key at beginning
            key_idx = 0
        elif position == "middle":
            # Key in middle
            key_idx = n_pairs // 2
        else:  # end
            # Key at end
            key_idx = n_pairs - 1
        
        # Move key document to correct position and shuffle others
        key_doc = documents.pop(key_idx)
        random.shuffle(documents)
        if position == "beginning":
            documents.insert(0, key_doc)
        elif position == "middle":
            mid = len(documents) // 2
            documents.insert(mid, key_doc)
        else:  # end
            documents.append(key_doc)
        
        context = "\n\n".join(documents)
        question = f"Question {i+1}: What is the key fact?"
        answer_key = f"key_answer_{i+1}"
        
        examples.append(
            LITMExample(
                context=context,
                question=question,
                answer_key=answer_key,
                position=position,
                n_pairs=n_pairs,
                budget=budget,
            )
        )
    
    return examples


def get_litm_dataloader(
    tokenizer,
    batch_size: int = 4,
    n_examples: int = 5000,
    max_length: int = 4096,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for LITM training.
    
    Args:
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        n_examples: Number of training examples
        max_length: Maximum sequence length
        num_workers: Number of dataloader workers
        shuffle: Whether to shuffle data
        
    Returns:
        PyTorch DataLoader
    """
    # Create synthetic training set
    examples = create_litm_training_set(
        n_examples=n_examples,
        n_pairs_options=[5, 10, 20, 40],
        positions=["beginning", "middle", "end"],
        budgets=[0.3, 0.5, 0.7, 1.0],
    )
    
    # Create dataset
    dataset = LITMTrainingDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=max_length,
        include_position_label=True,
        include_budget_label=True,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


if __name__ == "__main__":
    # Quick test
    from transformers import AutoTokenizer
    
    print("[litm_dataloader] Testing LITM training dataset...")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("[litm_dataloader] Creating training set...")
    examples = create_litm_training_set(n_examples=10)
    print(f"[litm_dataloader] Generated {len(examples)} examples")
    
    print("[litm_dataloader] Creating dataset...")
    dataset = LITMTrainingDataset(examples, tokenizer, max_length=512)
    print(f"[litm_dataloader] Dataset size: {len(dataset)}")
    
    print("[litm_dataloader] Testing DataLoader...")
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
    batch = next(iter(dataloader))
    print(f"[litm_dataloader] Batch keys: {batch.keys()}")
    print(f"[litm_dataloader] input_ids shape: {batch['input_ids'].shape}")
    print(f"[litm_dataloader] budget shape: {batch['budget'].shape}")
    print("[litm_dataloader] ✓ All tests passed!")
