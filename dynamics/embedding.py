

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
        self.no_action_emb = nn.Parameter(torch.randn(1, 1, embed_dim))
    
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

        return self.proj(actions)    #TODO Need to add learned bias for multi actiopn component
                                     #+ self.no_action_emb


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
            tau: (B,) continuous in [0, 1]
            d: (B,) step sizes (1, 0.5, 0.25, ..., 1/64)
        Returns:
            (B, 1, embed_dim)
        """

        tau_idx = tau * (self.num_tau_bins - 1).long()
        d_idx = (-d.log2()).long()

        tau_emb = self.tau_embedding(tau_idx)
        d_emb = self.d_embedding(d_idx)

        combined = torch.cat([tau_emb, d_emb], dim=-1)

        return combined.unsqueeze(1)

        
        
        