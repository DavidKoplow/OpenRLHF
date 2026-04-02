"""ES Worker Extension for vLLM.

Extends WorkerWrap with Evolutionary Strategies (ES) capabilities:
- model_mutate: Apply deterministic noise perturbations to model parameters
- apply_es_gradient: Compute and apply ES gradient from seed/score pairs
- get_mutation_seed: Return current mutation seed for response tagging
"""



import hashlib
import json
import ast
import os
from typing import List, Tuple, Optional

import torch

from .vllm_worker_wrap import WorkerWrap




import hashlib
import json
import ast
import os
from typing import List, Tuple, Optional

import torch

from .vllm_worker_wrap import WorkerWrap

STABILIZE_SEED = -1

def _noise(p: torch.Tensor, name: str, seed: int) -> torch.Tensor:

    g = torch.Generator(device=p.device)
    h = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    g.manual_seed((seed ^ h) & 0xFFFFFFFF)
    n = torch.normal(0, 1, size=p.shape, device=p.device, generator=g, dtype=p.dtype)
  
    return n

class ESWorkerWrap(WorkerWrap):
    """vLLM worker extension with ES mutation and gradient support."""

    current_seed: Optional[int] = None
    current_std: float = 0.0
    optimizer: Optional[torch.optim.Optimizer] = None



    def revert_mutation(self, lazy: bool=False):
        if not lazy and isinstance(self.current_seed, int) and self.current_std:
            for n, p in self.model_runner.model.named_parameters():
                orig_dtype = p.data.dtype
                p.data.copy_(
                    (p.data.float() - _noise(p.data, n, self.current_seed).float() * self.current_std)
                    .to(orig_dtype)
                )
        self.current_seed, self.current_std = None, 0.0
        return True

    def apply_mutation(self, seed: Optional[int]=None, std: float=0.0) -> Optional[int]:
        if isinstance(seed, int) and std:
            for n, p in self.model_runner.model.named_parameters():
                orig_dtype = p.data.dtype
                p.data.copy_(
                    (p.data.float() + _noise(p.data, n, seed).float() * std)
                    .to(orig_dtype)
                )
        self.current_seed, self.current_std = seed, std
        return seed

    def update_weight(self, **kwargs):
        self.revert_mutation(lazy=True)
        super().update_weight(**kwargs)

    def update_weight_cuda_ipc(self, **kwargs):
        self.revert_mutation(lazy=True)
        super().update_weight_cuda_ipc(**kwargs)

    def model_mutate(self, seed: Optional[int]=None, std: float=0.0) -> Optional[int]:

        self.revert_mutation()
        if seed == STABILIZE_SEED:
            self.current_seed = STABILIZE_SEED
            self.current_std = 0.0
            return seed
        self.apply_mutation(seed, std)

        return seed

    def get_mutation_seed(self) -> Optional[int]:
        """Return current mutation seed for response tagging."""
        return self.current_seed
    
    def _get_or_create_optimizer(self) -> torch.optim.Optimizer:
        """Lazy-initialize optimizer from environment variables."""
        if self.optimizer is not None:
            return self.optimizer
        
        # Read optimizer config from environment
        optimizer_name = os.getenv("ES_OPTIMIZER", "SGD")
        optimizer_params_str = os.getenv("ES_OPTIMIZER_PARAMS", '{"lr": 0.001}')
        
        try:
            optimizer_params = json.loads(optimizer_params_str)
        except json.JSONDecodeError:
            optimizer_params = ast.literal_eval(optimizer_params_str)
        
        # Get model parameters
        params = [p for n, p in self.model_runner.model.named_parameters()]
        
        # Create optimizer
        optimizer_cls = getattr(torch.optim, optimizer_name)
        self.optimizer = optimizer_cls(params, **optimizer_params)
        
        return self.optimizer

    def apply_es_gradient(self, updates: List[Tuple[int, float, float]]) -> bool:
        """Compute and apply ES gradient from seed/score pairs.
        
        Args:
            updates: List of (seed, normalized_score, sigma) tuples.
                    Scores should already be normalized (zero mean, optionally unit std).
                    
        Returns:
            True on success.
        """
        # Convert from dataclass if needed
        updates = [(u.seed, u.score, u.sigma) if hasattr(u, "seed") else u for u in updates]
        
        # Revert any current mutation before applying gradient
        self.revert_mutation()


        if len(updates) > 1:
            sc = torch.tensor([x[1] for x in updates], dtype=torch.float32)
            sd = sc.std()
            sc = (sc - sc.mean()) / sd if sd > 1e-8 else (sc - sc.mean())
            updates = [(seed, float(v), sig) for (seed, _, sig), v in zip(updates, sc)]

        clip = float(os.getenv("ES_CLIP_GRAD_NORM", "0.0"))
        optimizer = self._get_or_create_optimizer()
        optimizer.zero_grad(set_to_none=True)

        for name, p in self.model_runner.model.named_parameters():
            
            # Move optimizer state to GPU if needed
            st = optimizer.state.get(p)
            if st:
                for k, v in list(st.items()):
                    if isinstance(v, torch.Tensor):
                        st[k] = v.to(p.device)

            # Compute ES gradient: weighted sum of noise vectors
            g = torch.zeros_like(p, dtype=torch.float32)
            for seed, w, _ in updates:
                if isinstance(seed, int) and seed != STABILIZE_SEED:
                    g.add_(_noise(p.data, name, seed).float(), alpha=float(w))
            g.div_(max(1, len(updates)))
            
            # ES uses negative gradient (we want to move in direction of positive scores)
            p.grad = (-g).to(p.dtype)
            
            if clip > 0:
                torch.nn.utils.clip_grad_norm_([p], clip)
            
            optimizer.step()
            p.grad = None

            # Move optimizer state back to CPU to save memory
            st = optimizer.state.get(p)
            if st:
                for k, v in list(st.items()):
                    if isinstance(v, torch.Tensor):
                        st[k] = v.to("cpu")

        self.current_seed, self.current_std = None, 0.0
        return True

