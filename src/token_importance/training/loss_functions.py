"""Phase B Training Loss Functions

Composite loss combining:
1. Task loss (language modeling on ground-truth answer)
2. KL distillation (student vs. teacher logits)
3. Budget loss (soft constraint on token count)
4. Churn loss (smooth importance decisions)
5. Saliency loss (encourage important tokens survive budget)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class TISCompositeLoss(torch.nn.Module):
    """Composite loss for Phase B training.
    
    Args:
        lambda_kl: Weight for KL distillation loss (default: 0.1)
        lambda_budget: Weight for budget constraint loss (default: 0.01)
        lambda_churn: Weight for churn minimization loss (default: 0.01)
        lambda_saliency: Weight for saliency preservation loss (default: 0.0)
        kl_temperature: Temperature for softening KL targets (default: 2.0)
    """
    
    def __init__(
        self,
        lambda_kl: float = 0.1,
        lambda_budget: float = 0.01,
        lambda_churn: float = 0.01,
        lambda_saliency: float = 0.0,
        kl_temperature: float = 2.0,
    ):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_budget = lambda_budget
        self.lambda_churn = lambda_churn
        self.lambda_saliency = lambda_saliency
        self.kl_temperature = kl_temperature
    
    def forward(
        self,
        student_logits: torch.Tensor,
        target_ids: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        keep_masks: Optional[torch.Tensor] = None,
        importance_scores: Optional[torch.Tensor] = None,
        budget_tokens: Optional[int] = None,
        actual_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            student_logits: [batch_size, seq_len, vocab_size]
            target_ids: [batch_size, seq_len] ground-truth token IDs
            teacher_logits: [batch_size, seq_len, vocab_size] frozen teacher logits
            keep_masks: [num_updates, seq_len] binary masks from re-scoring cycles
            importance_scores: [seq_len] learned importance scores [0, 1]
            budget_tokens: Maximum tokens allowed (soft constraint)
            actual_tokens: Actual tokens used in generation
            
        Returns:
            Dict with keys:
                - loss_total: Composite loss
                - loss_task: Task loss (cross-entropy)
                - loss_kl: KL divergence
                - loss_budget: Budget violation penalty
                - loss_churn: Importance oscillation penalty
                - loss_saliency: Saliency preservation penalty
        """
        losses = {}
        
        # 1. Task Loss: language modeling on ground truth
        loss_task = self._compute_task_loss(student_logits, target_ids)
        losses["loss_task"] = loss_task
        
        # 2. KL Distillation Loss
        loss_kl = torch.tensor(0.0, device=student_logits.device)
        if teacher_logits is not None and self.lambda_kl > 0:
            loss_kl = self._compute_kl_loss(student_logits, teacher_logits)
        losses["loss_kl"] = loss_kl
        
        # 3. Budget Loss
        loss_budget = torch.tensor(0.0, device=student_logits.device)
        if budget_tokens is not None and actual_tokens is not None and self.lambda_budget > 0:
            loss_budget = self._compute_budget_loss(budget_tokens, actual_tokens)
        losses["loss_budget"] = loss_budget
        
        # 4. Churn Loss
        loss_churn = torch.tensor(0.0, device=student_logits.device)
        if keep_masks is not None and self.lambda_churn > 0:
            loss_churn = self._compute_churn_loss(keep_masks)
        losses["loss_churn"] = loss_churn
        
        # 5. Saliency Loss
        loss_saliency = torch.tensor(0.0, device=student_logits.device)
        if (importance_scores is not None and keep_masks is not None and 
            self.lambda_saliency > 0):
            loss_saliency = self._compute_saliency_loss(importance_scores, keep_masks)
        losses["loss_saliency"] = loss_saliency
        
        # Composite loss
        loss_total = (
            loss_task +
            self.lambda_kl * loss_kl +
            self.lambda_budget * loss_budget +
            self.lambda_churn * loss_churn +
            self.lambda_saliency * loss_saliency
        )
        losses["loss_total"] = loss_total
        
        return losses
    
    def _compute_task_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss on ground-truth tokens.
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            target_ids: [batch_size, seq_len]
            
        Returns:
            Scalar loss tensor
        """
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        target_ids_flat = target_ids.reshape(-1)
        loss = F.cross_entropy(logits_flat, target_ids_flat, reduction="mean")
        return loss
    
    def _compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence between student and teacher distributions.
        
        Args:
            student_logits: [batch_size, seq_len, vocab_size]
            teacher_logits: [batch_size, seq_len, vocab_size]
            
        Returns:
            Scalar KL loss
        """
        # Soften distributions with temperature
        student_log_probs = F.log_softmax(student_logits / self.kl_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.kl_temperature, dim=-1)
        
        # KL(teacher || student)
        kl_div = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs), dim=-1)
        
        # Average over sequence and batch
        return kl_div.mean()
    
    def _compute_budget_loss(
        self,
        budget_tokens: int,
        actual_tokens: int,
    ) -> torch.Tensor:
        """Soft penalty for exceeding token budget.
        
        Args:
            budget_tokens: Allocated budget
            actual_tokens: Actual tokens used
            
        Returns:
            Scalar budget loss (0 if within budget, squared overage otherwise)
        """
        overage = max(0, actual_tokens - budget_tokens)
        loss = torch.tensor(overage ** 2, dtype=torch.float32)
        return loss / (budget_tokens ** 2 + 1)  # Normalize by budget
    
    def _compute_churn_loss(
        self,
        keep_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Hamming distance between consecutive re-scoring masks.
        
        Args:
            keep_masks: [num_updates, seq_len] binary masks
            
        Returns:
            Scalar churn loss (average Hamming distance)
        """
        if keep_masks.shape[0] < 2:
            return torch.tensor(0.0, device=keep_masks.device)
        
        # Compute Hamming distance between consecutive masks
        mask_diffs = torch.abs(
            keep_masks[1:].float() - keep_masks[:-1].float()
        )  # [num_updates-1, seq_len]
        
        # Average Hamming distance per update
        churn_per_update = mask_diffs.sum(dim=1) / keep_masks.shape[1]
        
        # Return mean churn across all updates
        return churn_per_update.mean()
    
    def _compute_saliency_loss(
        self,
        importance_scores: torch.Tensor,
        keep_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Encourage high-importance tokens to be retained.
        
        Args:
            importance_scores: [seq_len] scores in [0, 1]
            keep_masks: [num_updates, seq_len] binary masks (averaged)
            
        Returns:
            Scalar saliency loss (negative correlation)
        """
        # Average retention across updates
        avg_retention = keep_masks.float().mean(dim=0)  # [seq_len]
        
        # Loss = -E[score_i * retention_i]
        # Encourages high scores to correlate with high retention
        loss = -(importance_scores * avg_retention).mean()
        
        return loss


class KLDivergenceLoss(torch.nn.Module):
    """Standalone KL divergence loss for teacher-student alignment."""
    
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: [batch_size, seq_len, vocab_size]
            teacher_logits: [batch_size, seq_len, vocab_size]
            
        Returns:
            Scalar KL loss
        """
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kl_div = torch.sum(
            teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs),
            dim=-1
        )
        
        return kl_div.mean()


class BudgetConstraintLoss(torch.nn.Module):
    """Soft penalty for token budget violations."""
    
    def forward(self, budget_tokens: int, actual_tokens: int) -> torch.Tensor:
        """
        Args:
            budget_tokens: Allocated token budget
            actual_tokens: Actual tokens used
            
        Returns:
            Scalar loss (0 if within budget)
        """
        overage = max(0, actual_tokens - budget_tokens)
        loss = torch.tensor(overage ** 2, dtype=torch.float32)
        return loss / (budget_tokens ** 2 + 1)


class ChurnLoss(torch.nn.Module):
    """Minimize oscillation in importance decisions."""
    
    def forward(self, keep_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keep_masks: [num_updates, seq_len] binary masks
            
        Returns:
            Scalar churn loss
        """
        if keep_masks.shape[0] < 2:
            return torch.tensor(0.0, device=keep_masks.device)
        
        # Hamming distance between consecutive masks
        mask_diffs = torch.abs(
            keep_masks[1:].float() - keep_masks[:-1].float()
        )
        
        churn_per_update = mask_diffs.sum(dim=1) / keep_masks.shape[1]
        return churn_per_update.mean()


class SaliencyLoss(torch.nn.Module):
    """Encourage high-importance tokens to survive budget cuts."""
    
    def forward(
        self,
        importance_scores: torch.Tensor,
        keep_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            importance_scores: [seq_len] scores in [0, 1]
            keep_masks: [num_updates, seq_len] binary masks
            
        Returns:
            Scalar saliency loss
        """
        avg_retention = keep_masks.float().mean(dim=0)
        loss = -(importance_scores * avg_retention).mean()
        return loss
