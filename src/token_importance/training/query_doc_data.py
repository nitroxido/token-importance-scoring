"""Create synthetic query-document dataset for Phase 4 pilot testing.

Uses NarrativeQA to create (query, relevant_doc, hard_negatives) triplets.
This allows testing the query-aware architecture without downloading GBs of MS MARCO.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from token_importance.training.data import load_training_dataset, extract_fields


class SyntheticQueryDocDataset(Dataset):
    """Synthetic query-document triplets from NarrativeQA.
    
    Creates training examples:
        - query: Question from NarrativeQA
        - positive_doc: Relevant passage (split from full document)
        - negative_docs: Random passages from other documents
    
    This mimics MS MARCO structure but uses existing NarrativeQA data.
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        split: str = "train",
        max_samples: int | None = None,
        max_doc_length: int = 512,
        num_hard_negatives: int = 3,
        doc_chunk_size: int = 400,  # Split long docs into chunks
        seed: int = 42,
    ):
        """Initialize synthetic dataset.
        
        Args:
            tokenizer: HF tokenizer for the base model
            split: Dataset split ("train" or "validation")
            max_samples: Limit number of samples
            max_doc_length: Max tokens per document chunk
            num_hard_negatives: Number of negative docs per query
            doc_chunk_size: Tokens per document chunk
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.max_doc_length = max_doc_length
        self.num_hard_negatives = num_hard_negatives
        self.doc_chunk_size = doc_chunk_size
        
        random.seed(seed)
        
        # Load NarrativeQA
        print(f"[synthetic] Loading NarrativeQA {split}...")
        dataset = load_training_dataset("narrativeqa", split=split, max_samples=max_samples)
        
        # Extract and process examples
        self.examples = []
        self.doc_chunks = []  # Pool of document chunks for hard negatives
        
        for item in dataset:
            fields = extract_fields(item, "narrativeqa")
            if fields is None:
                continue
            
            passage, question, answer = fields
            
            # Split passage into chunks
            chunks = self._split_into_chunks(passage)
            if not chunks:
                continue
            
            # Find chunk containing answer (positive)
            pos_chunk = self._find_answer_chunk(chunks, answer)
            
            self.examples.append({
                'query': question,
                'positive_doc': pos_chunk,
                'answer': answer,
            })
            
            # Add all chunks to pool (for sampling negatives)
            self.doc_chunks.extend(chunks)
        
        print(f"[synthetic] Created {len(self.examples)} examples")
        print(f"[synthetic] Document chunk pool: {len(self.doc_chunks)} chunks")
    
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
        """Find chunk containing answer. If not found, return first chunk."""
        answer_lower = answer.lower()
        for chunk in chunks:
            if answer_lower in chunk.lower():
                return chunk
        return chunks[0]  # Fallback
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a training example.
        
        Returns:
            dict with keys:
                - 'query': Question text
                - 'positive_doc': Relevant document chunk
                - 'negative_docs': List of N hard negative chunks
        """
        ex = self.examples[idx]
        
        # Sample hard negatives (random chunks from pool)
        # Filter out the positive doc to avoid duplicates
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
    """Collate batch of query-doc examples into tensors.
    
    Args:
        batch: List of examples from SyntheticQueryDocDataset
        tokenizer: HF tokenizer
        max_length: Max total sequence length
    
    Returns:
        dict with:
            - 'input_ids': [B, T] concatenated [query | doc1 | doc2 | ... | docN]
            - 'attention_mask': [B, T]
            - 'query_mask': [B, T] (1 for query tokens, 0 elsewhere)
            - 'doc_masks': [B, N, T] (1 for doc_i tokens, 0 elsewhere)
            - 'labels': [B] (always 0 for first doc = positive)
    """
    batch_size = len(batch)
    num_docs = 1 + len(batch[0]['negative_docs'])  # 1 positive + N negatives
    
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
            max_length=128,  # Limit query length
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
                max_length=400,  # Limit doc length
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
            input_ids += [tokenizer.eos_token_id] + doc_ids  # SEP + doc
            doc_start = len(input_ids) - len(doc_ids)
            doc_end = len(input_ids)
            doc_starts.append(doc_start)
            doc_ends.append(doc_end)
        
        input_ids.append(tokenizer.eos_token_id)  # Final EOS
        
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
    
    # Convert to tensors
    return {
        'input_ids': torch.tensor(input_ids_padded, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask_padded, dtype=torch.long),
        'query_mask': torch.tensor(query_mask_padded, dtype=torch.long),
        'doc_masks': torch.tensor(doc_masks_padded, dtype=torch.long),  # [B, N, T]
        'labels': torch.zeros(batch_size, dtype=torch.long),  # Always 0 (first doc is positive)
    }
