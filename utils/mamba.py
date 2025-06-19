import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from einops import rearrange, repeat, einsum

def generate(model, tokenizer, prompt: str, n_tokens_to_gen=50, sample=True, top_k=40):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    yield tokenizer.decode(input_ids[0])  # Yield the initial prompt

    for _ in range(n_tokens_to_gen):
        with torch.no_grad():
            logits = model(input_ids)[:, -1]

        probs = F.softmax(logits, dim=-1)
        if top_k is not None:
            top_values, _ = torch.topk(probs, k=top_k)
            probs[probs < top_values[:, -1, None]] = 0
            probs /= probs.sum(dim=-1, keepdim=True)

        if sample:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        yield tokenizer.decode(next_token[0])


def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            (args, state_dict): Tuple of model arguments and state_dict.
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = {
            "d_model": config_data['d_model'],
            "n_layer": config_data['n_layer'],
            "vocab_size": config_data['vocab_size']
        }
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        
        return args, new_state_dict


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
    

def ssm(x, A_log, D, x_proj, dt_rank, dt_proj):
    """Runs the SSM. See:
        - Algorithm 2 in Section 3.2 in the Mamba paper [1]
        - run_SSM(A, B, C, u) in The Annotated S4 [2]

    Args:
        x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
        A_log: ?
        D: ?
        x_proj: ?
        dt_rank: ?
        dt_proj: ?

    Returns:
        output: shape (b, l, d_in)

    Official Implementation:
        mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        
    """
    (d_in, n) = A_log.shape

    # Compute ∆ A B C D, the state space parameters.
    #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    #                                  and is why Mamba is called **selective** state spaces)
    
    A = -torch.exp(A_log.float())  # shape (d_in, n)
    D = D.float()

    x_dbl = x_proj(x)  # (b, l, dt_rank + 2*n)
    
    (delta, B, C) = x_dbl.split(split_size=[dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
    delta = F.softplus(dt_proj(delta))  # (b, l, d_in)
    
    y = selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
    
    return y

    
def selective_scan(u, delta, A, B, C, D):
    """Does selective scan algorithm. See:
        - Section 2 State Space Models in the Mamba paper [1]
        - Algorithm 2 in Section 3.2 in the Mamba paper [1]
        - run_SSM(A, B, C, u) in The Annotated S4 [2]

    This is the classic discrete state space formula:
        x(t + 1) = Ax(t) + Bu(t)
        y(t)     = Cx(t) + Du(t)
    except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

    Args:
        u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
        delta: shape (b, l, d_in)
        A: shape (d_in, n)
        B: shape (b, l, n)
        C: shape (b, l, n)
        D: shape (d_in,)

    Returns:
        output: shape (b, l, d_in)

    Official Implementation:
        selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
        Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
        
    """
    (b, l, d_in) = u.shape
    n = A.shape[1]
    
    # Discretize continuous parameters (A, B)
    # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
    # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
    #   "A is the more important term and the performance doesn't change much with the simplification on B"
    deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
    deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
    
    # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
    # Note that the below is sequential, while the official implementation does a much faster parallel scan that
    # is additionally hardware-aware (like FlashAttention).
    x = torch.zeros((b, d_in, n), device=deltaA.device)
    ys = []    
    for i in range(l):
        x = deltaA[:, i] * x + deltaB_u[:, i]
        y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
        ys.append(y)
    y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
    
    y = y + u * D

    return y