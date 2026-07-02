"""Flexible dataset loader supporting multiple sources.

Supports:
- NarrativeQA: Synthetic triplets from document chunks
- MS MARCO: Real query-passage pairs with BM25 negatives
- Custom: Any dataset with (query, positive_passages, negative_passages) structure
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Literal

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from token_importance.training.data import load_training_dataset, extract_fields


class FlexibleQueryDocDataset(Dataset):
    """Universal query-document dataset supporting multiple sources.
    
    Supports:
    - 'narrativeqa': NarrativeQA (synthetic chunks)
    - 'msmarco': MS MARCO (real passages)
    - Auto-detect from directory structure
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_name: str = "narrativeqa",
        split: str = "train",
        data_dir: Optional[str] = None,
        max_samples: int | None = None,
        max_doc_length: int = 512,
        num_hard_negatives: int = 3,
        doc_chunk_size: int = 400,
        seed: int = 42,
    ):
        """Initialize flexible dataset.
        
        Args:
            tokenizer: HF tokenizer
            dataset_name: 'narrativeqa', 'msmarco', or auto-detect from data_dir
            split: 'train' or 'validation'
            data_dir: Directory with pre-downloaded MS MARCO (e.g., data/msmarco_quick)
            max_samples: Limit number of samples
            max_doc_length: Max tokens per document
            num_hard_negatives: Number of negative passages
            doc_chunk_size: Tokens per chunk
            seed: Random seed
        """
        self.tokenizer = tokenizer
        self.max_doc_length = max_doc_length
        self.num_hard_negatives = num_hard_negatives
        self.doc_chunk_size = doc_chunk_size
        self.dataset_name = dataset_name
        
        random.seed(seed)
        
        # Auto-detect dataset type from data_dir if provided
        if data_dir:
            data_path = Path(data_dir)
            if data_path.exists():
                if "msmarco" in str(data_path):
                    dataset_name = "msmarco"
                elif "narrativeqa" in str(data_path):
                    dataset_name = "narrativeqa"
        
        self.dataset_name = dataset_name
        
        # Load dataset based on type
        if dataset_name.lower() == "msmarco":
            self._load_msmarco(data_dir, split, max_samples)
        else:  # Default to NarrativeQA
            self._load_narrativeqa(split, max_samples)
    
    def _load_narrativeqa(self, split: str, max_samples: int | None):
        """Load NarrativeQA synthetic dataset."""
        print(f"[dataset] Loading NarrativeQA {split}...")
        dataset = load_training_dataset("narrativeqa", split=split, max_samples=max_samples)
        
        self.examples = []
        self.doc_chunks = []
        
        for item in dataset:
            fields = extract_fields(item, "narrativeqa")
            if fields is None:
                continue
            
            passage, question, answer = fields
            chunks = self._split_into_chunks(passage)
            if not chunks:
                continue
            
            pos_chunk = self._find_answer_chunk(chunks, answer)
            self.examples.append({
                'query': question,
                'positive_doc': pos_chunk,
                'answer': answer,
            })
            self.doc_chunks.extend(chunks)
        
        print(f"[dataset] NarrativeQA: {len(self.examples)} examples, {len(self.doc_chunks)} chunks")
    
    def _load_msmarco(self, data_dir: Optional[str], split: str, max_samples: int | None):
        """Load MS MARCO dataset from local directory."""
        if not data_dir:
            data_dir = "data/msmarco_quick"
        
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"MS MARCO directory not found: {data_dir}")
        
        print(f"[dataset] Loading MS MARCO from {data_dir}...")
        
        # Load pre-saved dataset using HF datasets library
        from datasets import load_from_disk
        
        try:
            dataset = load_from_disk(str(data_path / split))
        except Exception as e:
            raise ValueError(f"Could not load MS MARCO from {data_path / split}: {e}")
        
        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))
        
        self.examples = []
        self.doc_chunks = []
        
        for item in dataset:
            # MS MARCO format: query, passages (dict with passage_text list)
            query = item.get('query', '')
            if not query:
                continue
            
            passages_data = item.get('passages', {})
            passage_texts = passages_data.get('passage_text', []) if isinstance(passages_data, dict) else []
            
            if not passage_texts:
                continue
            
            # Use first passage as positive, others as potential negatives
            positive_passage = passage_texts[0]
            negative_passages = passage_texts[1:] if len(passage_texts) > 1 else []
            
            self.examples.append({
                'query': query,
                'positive_doc': positive_passage,
                'negative_docs': negative_passages[:self.num_hard_negatives],
            })
            
            # Add all passages to chunk pool
            self.doc_chunks.extend(passage_texts)
        
        print(f"[dataset] MS MARCO: {len(self.examples)} examples, {len(self.doc_chunks)} passages")
    
    def _split_into_chunks(self, text: str) -> list[str]:
        """Split long text into token-limited chunks."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), self.doc_chunk_size):
            chunk_tokens = tokens[i:i + self.doc_chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        return chunks
    
    def _find_answer_chunk(self, chunks: list[str], answer: str) -> str:
        """Find chunk containing answer."""
        answer_lower = answer.lower()
        for chunk in chunks:
            if answer_lower in chunk.lower():
                return chunk
        return chunks[0]
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> dict:
        """Get training example."""
        ex = self.examples[idx]
        
        # Handle both synthetic (with doc_chunks pool) and real (with negative_docs) formats
        if 'negative_docs' in ex:
            # MS MARCO format: use provided negatives
            negatives = ex['negative_docs']
            if len(negatives) < self.num_hard_negatives:
                # Sample from chunk pool if not enough provided
                neg_pool = [c for c in self.doc_chunks if c != ex['positive_doc']]
                additional = random.sample(
                    neg_pool, 
                    min(self.num_hard_negatives - len(negatives), len(neg_pool))
                )
                negatives = negatives + additional
        else:
            # NarrativeQA format: sample from chunk pool
            neg_pool = [c for c in self.doc_chunks if c != ex['positive_doc']]
            negatives = random.sample(neg_pool, min(self.num_hard_negatives, len(neg_pool)))
        
        return {
            'query': ex['query'],
            'positive_doc': ex['positive_doc'],
            'negative_docs': negatives,
        }


def collate_query_doc_batch(
    batch: list[dict],
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
) -> dict:
    """Collate batch of query-doc examples into tensors."""
    batch_size = len(batch)
    num_docs = 1 + len(batch[0]['negative_docs'])
    
    input_ids_batch = []
    attention_mask_batch = []
    query_mask_batch = []
    doc_masks_batch = []
    
    for ex in batch:
        # Tokenize query
        query_enc = tokenizer(
            ex['query'],
            add_special_tokens=False,
            truncation=True,
            max_length=128,
        )
        query_ids = query_enc['input_ids']
        
        # Tokenize documents (positive first, then negatives)
        docs = [ex['positive_doc']] + ex['negative_docs']
        doc_ids_list = []
        for doc in docs:
            doc_enc = tokenizer(
                doc,
                add_special_tokens=False,
                truncation=True,
                max_length=400,
            )
            doc_ids_list.append(doc_enc['input_ids'])
        
        # Concatenate: [BOS] query [SEP] doc1 [SEP] doc2 [SEP] ... [EOS]
        input_ids = [tokenizer.bos_token_id] + query_ids
        
        # Track positions
        query_start = 1
        query_end = len(input_ids)
        doc_starts = []
        doc_ends = []
        
        for doc_ids in doc_ids_list:
            input_ids += [tokenizer.eos_token_id] + doc_ids
            doc_start = len(input_ids) - len(doc_ids)
            doc_end = len(input_ids)
            doc_starts.append(doc_start)
            doc_ends.append(doc_end)
        
        input_ids.append(tokenizer.eos_token_id)
        
        # Truncate if too long
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        seq_len = len(input_ids)
        
        # Create masks
        attention_mask = [1] * seq_len
        query_mask = [0] * seq_len
        query_mask[query_start:min(query_end, seq_len)] = [1] * (min(query_end, seq_len) - query_start)
        
        doc_masks = []
        for doc_start, doc_end in zip(doc_starts, doc_ends):
            doc_mask = [0] * seq_len
            if doc_start < seq_len:
                actual_end = min(doc_end, seq_len)
                doc_mask[doc_start:actual_end] = [1] * (actual_end - doc_start)
            doc_masks.append(doc_mask)
        
        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        query_mask_batch.append(query_mask)
        doc_masks_batch.append(doc_masks)
    
    # Pad to max length in batch
    max_batch_len = max(len(ids) for ids in input_ids_batch)
    
    def pad_sequence(seq, max_len, pad_value=0):
        return seq + [pad_value] * (max_len - len(seq))
    
    input_ids_padded = [pad_sequence(ids, max_batch_len, tokenizer.pad_token_id) 
                        for ids in input_ids_batch]
    attention_mask_padded = [pad_sequence(mask, max_batch_len, 0) 
                             for mask in attention_mask_batch]
    query_mask_padded = [pad_sequence(mask, max_batch_len, 0) 
                         for mask in query_mask_batch]
    
    doc_masks_padded = []
    for doc_masks in doc_masks_batch:
        doc_masks_padded.append([pad_sequence(mask, max_batch_len, 0) for mask in doc_masks])
    
    return {
        'input_ids': torch.tensor(input_ids_padded, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask_padded, dtype=torch.long),
        'query_mask': torch.tensor(query_mask_padded, dtype=torch.long),
        'doc_masks': torch.tensor(doc_masks_padded, dtype=torch.long),
        'labels': torch.zeros(batch_size, dtype=torch.long),
    }
