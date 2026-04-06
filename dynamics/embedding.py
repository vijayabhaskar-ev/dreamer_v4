from typing import Optional

import torch
import torch.nn as nn


class AgentTokenEmbedding(nn.Module):
    """ The agent token is inserted into the dynamics transformer as a new
    modality.  It attends to all other tokens but is invisible to them
    (asymmetric spatial mask), preventing causal confusion.

    Single-task: one learned parameter.
    Multi-task:  nn.Embedding lookup from task_id.
    """

    def __init__(self, embed_dim: int, num_tasks: int = 1):
        super().__init__()
        self.num_tasks = num_tasks
        if num_tasks == 1:
            self.embedding = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02) #TODO Add the variance scaling as a config
        else:
            self.embedding = nn.Embedding(num_tasks, embed_dim)
            nn.init.normal_(self.embedding.weight, std=0.02)


    def forward(self, batch_size: int, task_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns (B, 1, embed_dim)."""
        if self.num_tasks == 1:
            return self.embedding.expand(batch_size, -1, -1)
        else:
            # task_id: (B,) int tensor
            return self.embedding(task_id).unsqueeze(1)  # (B, 1, D)


class ActionEmbedding(nn.Module):
    """
    Encodes actions into S_a=1 token.
    - Continuous actions: linear projection + learned bias
    - No action: just the learned embedding
    """
    def __init__(self, action_dim, embed_dim):
        super().__init__() 
        #TODO Need to add functionality for discrete actions /minecraft actions
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.proj = nn.Linear(action_dim, embed_dim)
        self.no_action_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
    
    def forward(self, actions, batch_size):
        """
        Args:
            actions: (B, 1, action_dim) or None
            batch_size: B (needed when actions is None)
        Returns:
            (B, 1, embed_dim)
        """
        
        if actions is None:
            assert batch_size is not None
            return self.no_action_emb.expand(batch_size, -1, -1)

        return self.proj(actions) + self.no_action_emb    


class TauDEmbedding(nn.Module):
    """
    Encodes (τ, d) into a single token.
    τ → discretize to bins → embedding lookup
    d → convert to index via -log2(d) → embedding lookup
    Concatenate both → (B, 1, embed_dim)
    """
    def __init__(self, embed_dim, num_tau_bins=256, num_d_bins=7):
        super().__init__()
        self.tau_embedding = nn.Embedding(num_tau_bins, embed_dim // 2)
        self.d_embedding = nn.Embedding(num_d_bins, embed_dim // 2)
        self.num_tau_bins = num_tau_bins
        self.num_d_bins = num_d_bins
    
    def forward(self, tau, d):
        """
        Args:
            tau: (B,T) continuous in [0, 1]
            d: (B,T) step sizes (1, 0.5, 0.25, ..., 1/64)
        Returns:
            (B, T, embed_dim)
        """

        tau_idx = (tau * (self.num_tau_bins - 1)).long().clamp(0, self.num_tau_bins - 1)
        d_idx = (-torch.log2(d)).long().clamp(0, self.num_d_bins - 1)

        tau_emb = self.tau_embedding(tau_idx)
        d_emb = self.d_embedding(d_idx)

        combined = torch.cat([tau_emb, d_emb], dim=-1) 

        return combined

        
        
        