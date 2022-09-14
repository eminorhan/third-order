import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def third_order_attention(query, key_0, key_1, value_0, value_1, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    # (B, nh, T, hs), (B, nh, T, hs), (B, nh, T, hs) -> (B, nh, T, T, T)
    scores = torch.einsum('bhkd,bhld,bhmd->bhklm', query, key_0, key_1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # (B, nh, T, hs) x (B, nh, T, hs) -> (B, nh, T, T, hs)   
    value = torch.einsum('bhkd,bhld->bhkld', value_0, value_1)

    # (B, nh, T, T, T) x (B, nh, T, T, hs) -> (B, nh, T, hs)   
    output = torch.einseum('bhklm,bhlmd->bhkd', p_attn, value)

    return output, p_attn


class ThirdOrderSelfAttention(nn.Module):
    def __init__(self, n_head, n_embd, dropout=0.1):
        "Take in model size and number of heads."
        super(ThirdOrderSelfAttention, self).__init__()

        # TODO: probably reparametrize in terms of n_head and n_embd // n_head (emb dim per head)
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd

        # projections
        self.W_q = nn.Linear(self.n_embd, self.n_embd)
        self.W_k_0 = nn.Linear(self.n_embd, self.n_embd)
        self.W_k_1 = nn.Linear(self.n_embd, self.n_embd)
        self.W_v_0 = nn.Linear(self.n_embd, self.n_embd)
        self.W_v_1 = nn.Linear(self.n_embd, self.n_embd)
        self.W_o = nn.Linear(self.n_embd, self.n_embd)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):
        
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        
        B, T, C = x.size()
        assert C == self.n_embd

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.W_q(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k_0 = self.W_k_0(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k_1 = self.W_k_1(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v_0 = self.W_v_0(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v_1 = self.W_v_1(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # apply attention on all the projected vectors in batch. 
        x, self.attn = third_order_attention(q, k_0, k_1, v_0, v_1, mask=mask, dropout=self.dropout)
        
        # concat using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(B, T, C)

        return self.W_o(x)