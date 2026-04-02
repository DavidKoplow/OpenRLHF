import asyncio
import os
import random
import time
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import SamplingParams

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.logging_utils import TensorboardLogger, WandbLogger, init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)

STABILIZE_SEED = -1


# =============================================================================
# ESExperience Dataclass
# =============================================================================


@dataclass
class ESExperience:
    """Experience dataclass for ES training.

    Simplified compared to PPO Experience - only contains fields needed for ES.

    Shapes:
        sequences: (B, S) token sequences
        attention_mask: (B, S) attention mask
        action_mask: (B, A) action mask
        rollout_log_probs: (B, S) optional log probs
        prompts: (B,) prompts for each sample
        labels: (B,) labels for each sample
        seeds: (B,) mutation seed for each sample
        rewards: (B,) rewards for each sample
    """

    sequences: torch.Tensor = None
    attention_mask: torch.LongTensor = None
    action_mask: torch.BoolTensor = None
    rollout_log_probs: torch.Tensor = None
    seeds: torch.Tensor = None
    rewards: torch.Tensor = None
    prompts: List[str] = None
    labels: List[str] = None


async def ray_get(x):
    return await asyncio.to_thread(ray.get, x)


async def run_one_seed(
    seed_idx: int,
    seed: int,
    *,
    engine_pool: asyncio.Queue,
    prompt_lock: asyncio.Lock,
    es_std,
    shared_batch: bool,
    num_seeds: int,
    generate_kwargs: dict,
    rollout_batch_size: int,
    get_prompts_for_seed,
    dispatch_and_collect,
    reward_model_group=None,
):
    engine = await engine_pool.get()
    try:
        await ray_get(engine.model_mutate.remote(seed, es_std))

        async with prompt_lock:
            prompts, labels, batch_exhausted = get_prompts_for_seed(
                seed_idx=seed_idx,
                num_seeds=num_seeds,
                num_prompts=rollout_batch_size,
                shared_batch=shared_batch,
            )

        round_exps = await asyncio.to_thread(
            dispatch_and_collect,
            [(prompts, labels, seed)],
            [engine],
            **generate_kwargs,
        )
    finally:
        engine_pool.put_nowait(engine)  # RELEASE ASAP

    reward_jobs = []
    if reward_model_group is not None and round_exps:
        # Batch all experiences from this seed into a single reward call
        all_sequences = [exp.sequences for exp in round_exps]
        all_attention_masks = [exp.attention_mask for exp in round_exps]
        refs = reward_model_group.async_run_method_batch(
            method_name="forward",
            sequences=all_sequences,
            attention_mask=all_attention_masks,
            pad_sequence=[True] * len(round_exps),
        )
        reward_jobs.append((round_exps, refs))  # (list_of_exps, list_of_refs) per seed

    return round_exps, batch_exhausted, len(prompts), reward_jobs


async def run_all_seeds(engine_seeds, vllm_engines, **kwargs):
    engine_pool = asyncio.Queue()
    for e in vllm_engines:
        engine_pool.put_nowait(e)

    prompt_lock = asyncio.Lock()

    tasks = [
        asyncio.create_task(
            run_one_seed(
                seed_idx=i,
                seed=seed,
                engine_pool=engine_pool,
                prompt_lock=prompt_lock,
                **kwargs,
            )
        )
        for i, seed in enumerate(engine_seeds)
    ]

    all_experiences = []
    all_reward_jobs = []
    exhausted = False
    prompts_consumed = 0

    for t in asyncio.as_completed(tasks):
        round_exps, batch_exhausted, consumed, reward_jobs = await t
        all_experiences.extend(round_exps)
        all_reward_jobs.extend(reward_jobs)
        exhausted |= batch_exhausted and not kwargs["shared_batch"]
        prompts_consumed += consumed

    if all_reward_jobs:
        reward_results = await asyncio.gather(*(ray_get(refs) for _, refs in all_reward_jobs))
        for (exps, _), rr in zip(all_reward_jobs, reward_results):
            if rr:
                # rr is list of lists from multiple actors - flatten it
                flat_rewards = sum(rr, [])
                for exp, reward in zip(exps, flat_rewards):
                    exp.rewards = reward[0] if isinstance(reward, list) else reward

    return all_experiences, exhausted, prompts_consumed


# =============================================================================
# Helper Functions
# =============================================================================


def _collect_prompt_batch(dataloader_iter, num_prompts: int):
    """Draw up to `num_prompts` items from the prompt dataloader.

    Args:
        dataloader_iter: Iterator over the dataloader
        num_prompts: Maximum number of prompts to collect

    Returns:
        Tuple of (prompts, labels, exhausted)
    """
    prompts, labels = [], []
    exhausted = False

    while len(prompts) < num_prompts:
        try:
            _, batch_prompts, batch_labels = next(dataloader_iter)
            remaining = num_prompts - len(prompts)
            prompts.extend(batch_prompts[:remaining])
            labels.extend(batch_labels[:remaining])
        except StopIteration:
            exhausted = True
            break

    return prompts, labels, exhausted


def prepare_datasets(strategy, tokenizer):
    """Prepare datasets for ES training.

    ES-specific version that doesn't require prompt_data_probs.

    Args:
        strategy: Training strategy with args
        tokenizer: HuggingFace tokenizer

    Returns:
        Tuple of (prompts_dataloader, eval_dataloader, max_steps)
    """
    args = strategy.args

    # Prepare train datasets - ES doesn't use probability blending
    train_data = blending_datasets(
        args.prompt_data,
        None,  # ES doesn't use prompt_data_probs
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.prompt_split,
    )

    # Create train dataset
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    prompts_dataset = PromptDataset(train_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, 1, True, True, prompts_dataset.collate_fn)

    # Create eval dataset if eval data exists
    if getattr(args, "eval_dataset", None):
        max_eval_samples = getattr(args, "max_eval_samples", None)
        if max_eval_samples is None:
            max_eval_samples = args.max_samples
        eval_data = blending_datasets(
            args.eval_dataset,
            None,  # No probability sampling for eval datasets
            strategy,
            args.seed,
            dataset_split=args.eval_split,
        )
        eval_data = eval_data.select(range(min(max_eval_samples, len(eval_data))))
        eval_dataset = PromptDataset(eval_data, tokenizer, strategy, input_template=args.input_template)
        eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False, eval_dataset.collate_fn)
    else:
        eval_dataloader = None

    # Calculate max steps for ES training
    n_samples_per_prompt = getattr(args, "n_samples_per_prompt", 1)
    max_epochs = getattr(args, "max_epochs", 1)
    max_steps = (
        len(prompts_dataset)
        * n_samples_per_prompt
        // args.rollout_batch_size
        * args.population_size
        * args.num_episodes
        * max_epochs
    )
    return prompts_dataloader, eval_dataloader, max_steps


# =============================================================================
# ESSamplesGenerator Class
# =============================================================================


class ESSamplesGenerator:
    """ES-specific sample generator with staged mutation and async reward computation.

    Key features:
    - Multi-round seed processing (handles seeds >> num_engines)
    - Configurable batch sharing (same vs unique data per seed)
    - Async reward computation per seed using reward_model_group
    """

    def __init__(
        self,
        strategy,
        prompts_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        tokenizer,
        vllm_engines: List,
        reward_model_group: Optional[RayActorGroup] = None,
    ):
        """Initialize ES samples generator.

        Args:
            strategy: Training strategy with args
            prompts_dataloader: DataLoader for training prompts
            eval_dataloader: DataLoader for evaluation prompts
            tokenizer: HuggingFace tokenizer
            vllm_engines: List of vLLM engine actors
            reward_model_group: Optional RayActorGroup for reward model (from BasePPOTrainer)
        """
        self.strategy = strategy
        self.args = strategy.args
        self.tokenizer = tokenizer
        self.vllm_engines = vllm_engines or []
        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader
        self.reward_model_group = reward_model_group

        # Internal state
        self._dataloader_iter = None
        self._eval_dataloader_iter = None
        self._cached_prompts = None
        self._cached_labels = None

    @torch.no_grad()
    def generate_samples(
        self,
        engine_seeds: List[int],
        es_std: float,
        shared_batch: bool = True,
        **generate_kwargs,
    ) -> Tuple[List[ESExperience], Optional[float], int, bool]:
        """Generate samples with ES mutations across multiple rounds.

        Args:
            engine_seeds: List of mutation seeds (can be >> num_engines)
            es_std: Standard deviation for ES perturbations
            shared_batch: If True, all seeds evaluate same prompts;
                         if False, each seed gets unique prompts via DistributedSampler
            **generate_kwargs: Generation parameters (temperature, max_new_tokens, etc.)

        Returns:
            Tuple of (experiences, filter_pass_rate, prompts_consumed, exhausted)
        """
        if self._dataloader_iter is None:
            self._dataloader_iter = iter(self.prompts_dataloader)

        # Wake sleeping vLLM engines before dispatching
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        # Clear cached batch at start of step
        self._cached_prompts = None
        self._cached_labels = None

        all_experiences, exhausted, prompts_consumed = asyncio.run(
            run_all_seeds(
                engine_seeds,
                self.vllm_engines,
                es_std=es_std,
                shared_batch=shared_batch,
                num_seeds=len(engine_seeds),
                rollout_batch_size=self.args.rollout_batch_size,
                get_prompts_for_seed=self._get_prompts_for_seed,
                dispatch_and_collect=self._dispatch_and_collect,
                reward_model_group=self.reward_model_group,
                generate_kwargs=generate_kwargs,
            )
        )
        filter_pass_rate = None
        return all_experiences, None, prompts_consumed, exhausted

    def _get_prompts_for_seed(
        self,
        seed_idx: int,
        num_seeds: int,
        num_prompts: int,
        shared_batch: bool,
    ) -> Tuple[List[str], List[str], bool]:
        """Get prompts for a specific seed.

        Args:
            seed_idx: Index of current seed in engine_seeds list
            num_seeds: Total number of seeds being processed
            num_prompts: Number of prompts to fetch
            shared_batch: If True, all seeds get same prompts; if False, each gets unique slice

        Returns:
            Tuple of (prompts, labels, exhausted)
        """
        if shared_batch:
            # All seeds get same prompts - use cached batch from first call
            if self._cached_prompts is None:
                self._cached_prompts, self._cached_labels, exhausted = _collect_prompt_batch(
                    self._dataloader_iter, num_prompts
                )
                self._cached_exhausted = exhausted
            return self._cached_prompts, self._cached_labels, getattr(self, "_cached_exhausted", False)
        else:
            # Each seed gets unique prompts via DistributedSampler
            # Use seed_idx as "rank" to partition data across seeds
            dataset = self.prompts_dataloader.dataset
            sampler = DistributedSampler(
                dataset,
                num_replicas=num_seeds,
                rank=seed_idx,
                seed=self.args.seed,
                shuffle=True,
            )

            # Create a temporary dataloader with this sampler
            temp_loader = DataLoader(
                dataset,
                batch_size=self.prompts_dataloader.batch_size,
                sampler=sampler,
                num_workers=0,
                collate_fn=getattr(self.prompts_dataloader, "collate_fn", None),
            )

            # Collect prompts
            prompts, labels = [], []
            exhausted = False
            for batch in temp_loader:
                if len(batch) >= 3:
                    _, batch_prompts, batch_labels = batch[:3]
                else:
                    batch_prompts = batch[0] if len(batch) > 0 else []
                    batch_labels = batch[1] if len(batch) > 1 else []

                remaining = num_prompts - len(prompts)
                if remaining <= 0:
                    break
                prompts.extend(batch_prompts[:remaining])
                labels.extend(batch_labels[:remaining])

            if len(prompts) < num_prompts:
                exhausted = True

            return prompts, labels, exhausted

    def _dispatch_and_collect(
        self,
        prompts_per_engine: List[Tuple[List[str], List[str], int]],
        engines: List,
        **generate_kwargs,
    ) -> List[ESExperience]:
        """Dispatch prompts to engines and collect responses.

        Uses batch generation to submit all prompts to vLLM concurrently,
        allowing vLLM's continuous batching to process them efficiently.

        Args:
            prompts_per_engine: List of (prompts, labels, seed) tuples per engine
            engines: List of vLLM engine actors to use
            **generate_kwargs: Generation parameters

        Returns:
            List of ESExperience objects
        """
        sampling_params = SamplingParams(
            temperature=generate_kwargs.get("temperature", 1.0),
            top_p=generate_kwargs.get("top_p", 1.0),
            top_k=generate_kwargs.get("top_k", -1),
            max_tokens=generate_kwargs.get("max_new_tokens", 1024),
            min_tokens=generate_kwargs.get("min_new_tokens", 1),
            skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
            logprobs=None,
            stop=[generate_kwargs.get("stop_token", "</answer>")],
        )
        truncate_length = generate_kwargs.get("prompt_max_len", 1024) + generate_kwargs.get("max_new_tokens", 1024)
        n_samples = self.args.n_samples_per_prompt if hasattr(self.args, "n_samples_per_prompt") else 1

        # Dispatch all prompts to their assigned engines using batch generation
        # Each engine receives all its prompts in a single Ray call
        all_refs = []  # (ref, seed, prompts, labels)
        total_samples = 0
        for engine_idx, (prompts, labels, seed) in enumerate(prompts_per_engine):
            engine = engines[engine_idx]
            # Single batch call per engine - vLLM will batch internally via continuous batching
            ref = engine.generate_responses_batch.remote(
                prompts=prompts,
                labels=labels,
                sampling_params=sampling_params,
                max_length=truncate_length,
                hf_tokenizer=self.tokenizer,
                num_samples=n_samples,
            )
            all_refs.append((ref, seed, prompts, labels))
            total_samples += len(prompts) * max(int(n_samples), 1)

        # Collect all responses
        experiences = []
        pbar = tqdm(total=total_samples, desc="Generate samples")

        pending = list(all_refs)
        while pending:
            # Wait for any result
            refs_only = [r[0] for r in pending]
            ready_refs, _ = ray.wait(refs_only, num_returns=1, timeout=10.0)

            for ready_ref in ready_refs:
                # Find the matching entry
                for i, (ref, seed, prompts, labels) in enumerate(pending):
                    if ref == ready_ref:
                        # Get batch of responses
                        batch_responses = ray.get(ready_ref)

                        # Process all responses from this batch
                        for response in batch_responses:
                            exp = self._process_response_into_experience(response, seed, truncate_length)
                            experiences.append(exp)

                        pending.pop(i)
                        pbar.update(len(batch_responses))
                        break

        pbar.close()
        return experiences

    def _process_response_into_experience(
        self,
        response: dict,
        seed: int,
        truncate_length: int,
    ) -> ESExperience:
        """Turn a single vLLM response into an ESExperience.

        Args:
            response: Response dict from vLLM engine
            seed: Mutation seed used for this generation
            truncate_length: Max sequence length

        Returns:
            ESExperience object
        """
        # Base rollout fields from the output
        tokenized_observation = response["observation_tokens"].copy()
        tokenized_ranges = response["action_ranges"]
        reward_val = response.get("reward", None)

        # Handle tensor values from reward functions (extract scalar)
        if isinstance(reward_val, torch.Tensor):
            reward_val = reward_val.mean().item()

        sequences = torch.tensor(tokenized_observation, dtype=torch.long)
        attention_mask = torch.tensor([1] * len(tokenized_observation))

        # Mark the action span within the concatenated tokens
        action_mask = torch.zeros_like(attention_mask)
        for start, end in tokenized_ranges:
            action_mask[start:end] = 1

        # Truncate everything to the configured context window
        sequences = sequences[:truncate_length].to("cpu")
        attention_mask = attention_mask[:truncate_length].to("cpu")
        action_mask = action_mask[1:truncate_length].to("cpu")

        # Align rollout logprobs with the truncated action span
        if response.get("rollout_log_probs") is not None:
            rollout_log_probs = torch.tensor(response["rollout_log_probs"][1:truncate_length]).to("cpu")
        else:
            rollout_log_probs = None

        return ESExperience(
            sequences=sequences.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
            seeds=torch.tensor([seed]),
            rewards=torch.tensor([reward_val]) if reward_val is not None else None,
            prompts=[response.get("prompt", "")],
            labels=[response.get("label", "")],
        )

    @torch.no_grad()
    def generate_eval_samples(self, **generate_kwargs) -> List[ESExperience]:
        """Generate evaluation samples without mutations (base model).

        Args:
            **generate_kwargs: Generation parameters

        Returns:
            List of ESExperience objects
        """
        if self._eval_dataloader_iter is None:
            self._eval_dataloader_iter = iter(self.eval_dataloader)

        # Wake sleeping vLLM engines before dispatching
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        # Revert any mutations on engines (use seed=None, std=0)
        revert_refs = [engine.model_mutate.remote(None, 0.0) for engine in self.vllm_engines]
        ray.get(revert_refs)

        # Collect all eval prompts
        prompts, labels, _ = _collect_prompt_batch(self._eval_dataloader_iter, len(self.eval_dataloader.dataset))

        # Create dummy seed=0 for eval (no mutation)
        prompts_per_engine = []
        num_engines = len(self.vllm_engines)
        prompts_per_eng = len(prompts) // num_engines + 1

        for i in range(num_engines):
            start = i * prompts_per_eng
            end = min(start + prompts_per_eng, len(prompts))
            if start < len(prompts):
                prompts_per_engine.append((prompts[start:end], labels[start:end], 0))  # seed=0 for eval

        # Dispatch and collect
        experiences = self._dispatch_and_collect(prompts_per_engine, self.vllm_engines, **generate_kwargs)

        # Compute rewards if reward model available
        if self.reward_model_group is not None:
            for exp in experiences:
                reward_refs = self.reward_model_group.async_run_method_batch(
                    method_name="forward",
                    sequences=[exp.sequences],
                    attention_mask=[exp.attention_mask],
                    pad_sequence=[True],
                )
                rewards_results = ray.get(reward_refs)
                if rewards_results:
                    exp.rewards = rewards_results[0][0] if isinstance(rewards_results[0], list) else rewards_results[0]

        # Put engines back to sleep when enabled
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        self._eval_dataloader_iter = None

        return experiences


class BaseTrainer(ABC):

    def __init__(
        self,
        strategy: DeepspeedStrategy,
        reward_model_group: RayActorGroup,
        vllm_engines,
        tokenizer,
    ) -> None:
        self.strategy = strategy
        self.args = strategy.args
        self.reward_model_group = reward_model_group
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer

        # Tracking backends
        self.wandb_logger = WandbLogger(self.args) if self.args.use_wandb else None
        self.tensorboard_logger = TensorboardLogger(self.args) if self.args.use_tensorboard else None

    def fit(self):
        raise NotImplementedError("fit method is not implemented")

    def broadcast_to_vllm(self) -> None:
        """Broadcast actor weights to vLLM engines.

        When vllm_enable_sleep is enabled, we use fine-grained control:
        1. Wake up only weights (not KV cache) to minimize GPU memory during weight sync
        2. Broadcast weights from actor model to vLLM
        3. Keep vLLM in weights-only state; KV cache will be woken up later before generation

        This approach reduces peak GPU memory during gradient sync by avoiding
        simultaneous allocation of both weights and KV cache.
        """
        if self.args.vllm_enable_sleep:
            # Wake up only weights for weight sync (not KV cache)
            # This avoids allocating KV cache memory during weight update
            batch_vllm_engine_call(self.vllm_engines, "wake_up", tags=["weights"])

        ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

        # NOTE: We keep vLLM in weights-only state after weight sync.
        # KV cache will be woken up before generation in SamplesGenerator.

    def save_logs_and_checkpoints(self, global_step: int, logs_dict=None, client_states=None) -> None:
        logs_dict = logs_dict or {}
        if global_step % self.args.logging_steps == 0:
            if self.wandb_logger:
                self.wandb_logger.log_train(global_step, logs_dict)
            if self.tensorboard_logger:
                self.tensorboard_logger.log_train(global_step, logs_dict)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        client_states = client_states or {}
        if global_step % self.args.save_steps == 0:
            tag = f"global_step{global_step}"
            refs = self.actor_model_group.async_run_method(
                method_name="save_checkpoint", tag=tag, client_states=client_states
            )
            if self.critic_model_group is not None:
                refs.extend(self.critic_model_group.async_run_method(method_name="save_checkpoint", tag=tag))
            ray.get(refs)

    def init_checkpoint_states(self) -> Dict:
        ckpt_path = os.path.join(self.args.ckpt_path, "_actor")
        if self.args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            logger.info(f"checkpoint_states: {checkpoint_states}")
            return checkpoint_states
        return {
            "episode": 0,
            "global_step": 0,
            "total_consumed_prompts": 0,
            "data_loader_state_dict": {},
        }


# =============================================================================
# ESTrainer Class
# =============================================================================


@ray.remote
class ESTrainer(BaseTrainer):
    """
    Evolutionary Strategies Trainer extending BasePPOTrainer.

    Inherits:
        - Logging infrastructure (Wandb/Tensorboard)
        - Checkpointing logic
        - Actor/Critic group management (though ES primarily uses VLLM engines)
        - Data pipeline setup
    """

    def __init__(
        self,
        pretrain: str,
        strategy,
        actor_model_group,
        critic_model_group,
        reward_model_group,
        reference_model_group,
        vllm_engines: List,
        optim: Optimizer,
        **generate_kwargs,
    ) -> None:
        # 1. Setup Datasets and Tokenizer (Reusing logic usually found in PPOTrainer)
        if strategy.args.eval_steps == -1:
            strategy.args.eval_steps = float("inf")
        if strategy.args.save_steps == -1:
            strategy.args.save_steps = float("inf")

        self.optim = optim

        tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)

        # Prepare datasets (Using the helper function defined in your snippet)
        self.prompts_dataloader, self.eval_dataloader, self.max_steps = prepare_datasets(strategy, tokenizer)

        # 2. Initialize Base Class
        super().__init__(
            strategy,
            reward_model_group,
            vllm_engines,
            tokenizer,
        )

        # Store the additional model groups that ESTrainer needs
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reference_model_group = reference_model_group

        # 3. ES Specific Configuration
        self.generate_kwargs = generate_kwargs
        self.es_std = getattr(self.args, "es_std", 0.01)
        self.population_size = getattr(self.args, "population_size", len(vllm_engines))
        self.es_shared_batch = getattr(self.args, "es_shared_batch", True)
        self._rng = random.Random(self.args.seed)

        # 4. Initialize ES Sample Generator
        # Custom implementation with staged mutation and async reward computation
        self.samples_generator = ESSamplesGenerator(
            strategy=strategy,
            prompts_dataloader=self.prompts_dataloader,
            eval_dataloader=self.eval_dataloader,
            tokenizer=tokenizer,
            vllm_engines=vllm_engines,
            reward_model_group=self.reward_model_group,  # From BasePPOTrainer
        )

    def fit(self) -> None:
        """
        Override fit to handle the specific ES training loop while leveraging
        BasePPOTrainer's helper methods.
        """
        # Restore states (checkpointing logic reused from BasePPOTrainer utils if available,
        # or we use the custom init below)
        checkpoint_states = self.init_checkpoint_states()
        global_step = checkpoint_states["global_step"]

        # Sync VLLM with Actor weights at start
        if global_step > 0:
            self.broadcast_to_vllm()

        logger.info(f"Starting ES Training from step {global_step}...")

        # Run step-0 evaluation before any training (baseline measurement)
        if global_step == 0 and self.eval_dataloader:
            logger.info("Running step-0 baseline evaluation (before training)...")
            self.evaluate(global_step, **self.generate_kwargs)

        # Infinite loop over episodes (standard OpenRLHF pattern)
        for episode in range(checkpoint_states["episode"], self.args.num_episodes):

            # Use the sample generator's internal handling or manual tqdm
            while True:
                # --- The Core ES Step ---
                # We do not use self.experience_maker here because ES
                # injects noise *before* generation, not after.

                status, global_step, is_exhausted = self.train_step(global_step)

                # Logging
                if global_step % self.args.logging_steps == 0:
                    log_status = {k: v for k, v in status.items() if not k.startswith("generated")}
                    self.save_logs_and_checkpoints(global_step, log_status)
                    logger.info(f"Step {global_step}: reward={status.get('avg_reward', 0):.4f}")

                # Evaluation
                if global_step % self.args.eval_steps == 0 and self.eval_dataloader:
                    self.evaluate(global_step, **self.generate_kwargs)

                if is_exhausted:
                    break

        # Cleanup
        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()

    def train_step(self, global_step: int) -> Tuple[Dict, int, bool]:
        """
        Executes one ES optimization step.
        Returns: (status_dict, next_global_step, is_dataset_exhausted)
        """
        start_time = time.time()

        # 1. Generate Seeds for the Population
        # We assign specific random seeds to specific VLLM engines to create perturbations
        seeds = [self._rng.randint(0, 2**31 - 1) for _ in range(self.population_size)]

        if getattr(self.args, "es_stabilize_seed", False):
            seeds[0] = STABILIZE_SEED

        # 2. Generate samples with ES mutations (multi-round if seeds > engines)
        population_start_time = time.time()
        rollout_samples, filter_rate, prompts_consumed, is_exhausted = self.samples_generator.generate_samples(
            engine_seeds=seeds,
            es_std=self.es_std,
            shared_batch=self.es_shared_batch,
            **self.generate_kwargs,
        )
        population_time = time.time() - population_start_time
        logger.info(f"Step {global_step}: Population processing time: {population_time:.2f}s")

        # 3. Aggregate Scores per Seed (ES Gradient Calculation)
        # ESExperience uses seeds tensor instead of info dict
        seed_scores = defaultdict(list)
        total_reward = 0.0

        for sample in rollout_samples:
            # Get seeds and rewards from the ESExperience tensors
            sample_seeds = sample.seeds.tolist() if sample.seeds is not None else []
            sample_rewards = sample.rewards.tolist() if sample.rewards is not None else []

            # Handle case where rewards might be scalar
            if not isinstance(sample_rewards, list):
                sample_rewards = [sample_rewards]
            if not isinstance(sample_seeds, list):
                sample_seeds = [sample_seeds]

            for seed_val, reward_val in zip(sample_seeds, sample_rewards):
                seed_scores[seed_val].append(reward_val)
                total_reward += reward_val

        # 4. Compute Normalized Updates
        updates = self._normalize_seed_scores(seed_scores)

        # 5. Apply ES gradient to all engines
        update_refs = [engine.apply_es_gradient.remote(updates) for engine in self.vllm_engines]
        ray.get(update_refs)

        # Optionally: Sync back to Actor Group if you maintain a master copy there
        # ray.get(self.actor_model_group.async_run_method("step_from_remote_params", ...))

        num_samples = sum(len(sample.seeds) if sample.seeds is not None else 0 for sample in rollout_samples)
        status = {
            "avg_reward": total_reward / max(1, num_samples),
            "num_samples": num_samples,
            "num_seeds": len(seed_scores),
            "step_time": time.time() - start_time,
            "prompts_consumed": prompts_consumed,
        }

        return status, global_step + 1, is_exhausted

    def _normalize_seed_scores(self, seed_scores: Dict[int, List[float]]) -> List[Tuple[int, float, float]]:
        """Helper to normalize scores for ES update."""
        if not seed_scores:
            return []

        seed_means = {seed: np.mean(scores) for seed, scores in seed_scores.items()}
        scores_tensor = torch.tensor(list(seed_means.values()), dtype=torch.float32)

        # Standardize
        mean = scores_tensor.mean()
        std = scores_tensor.std()
        if std < 1e-8:
            std = 1.0

        normalized = (scores_tensor - mean) / std

        updates = []
        for (seed, _), norm_score in zip(seed_means.items(), normalized.tolist()):
            updates.append((seed, norm_score, self.es_std))
        return updates

    @torch.no_grad()
    def evaluate(self, global_step, **generate_kwargs):
        """Evaluate model performance on eval dataset."""
        start_time = time.time()
        logger.info(f"Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # First collect all prompts and labels to build datasource mapping
        prompt_to_datasource = {}
        for datasources, prompts, labels in self.eval_dataloader:
            for prompt, datasource in zip(prompts, datasources):
                prompt_to_datasource[prompt] = datasource

        # Generate evaluation samples (no mutations, base model)
        validation_start_time = time.time()
        samples_list = self.samples_generator.generate_eval_samples(**generate_kwargs)
        validation_time = time.time() - validation_start_time
        logger.info(f"Step {global_step}: Validation generation time: {validation_time:.2f}s")

        # Collect all prompts from samples
        all_prompts = sum([s.prompts for s in samples_list if s.prompts], [])

        n_samples_per_prompt = generate_kwargs.get("n_samples_per_prompt", 1)

        # Get rewards from ESExperience objects
        rewards_list = []
        for sample in samples_list:
            if sample.rewards is not None:
                # Handle both tensor and scalar rewards
                if isinstance(sample.rewards, torch.Tensor):
                    rewards_list.extend(sample.rewards.tolist())
                else:
                    rewards_list.append(sample.rewards)

        if not rewards_list:
            logger.warning("No rewards collected during evaluation")
            return

        # Reshape rewards to (num_prompts, n_samples_per_prompt)
        rewards = torch.tensor(rewards_list)
        if len(rewards) % n_samples_per_prompt == 0:
            rewards = rewards.reshape(-1, n_samples_per_prompt)
        else:
            # Handle case where rewards don't evenly divide
            rewards = rewards.unsqueeze(1)

        # Collect local statistics for each data source
        global_metrics = {}

        # Process rewards in chunks
        num_prompts = len(all_prompts) // n_samples_per_prompt if n_samples_per_prompt > 0 else len(all_prompts)
        for i in range(min(num_prompts, len(rewards))):
            # Get the original prompt
            prompt_idx = i * n_samples_per_prompt if n_samples_per_prompt > 1 else i
            if prompt_idx >= len(all_prompts):
                break

            original_prompt = all_prompts[prompt_idx]
            datasource = prompt_to_datasource.get(original_prompt, "unknown")

            if datasource not in global_metrics:
                global_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}

            # Get rewards for this chunk
            chunk_rewards = rewards[i] if i < len(rewards) else torch.tensor([0.0])

            # Calculate pass@k and pass@1
            if n_samples_per_prompt > 1 and len(chunk_rewards) > 1:
                global_metrics[datasource][f"pass{n_samples_per_prompt}"] += chunk_rewards.max().float().item()
            global_metrics[datasource]["pass1"] += chunk_rewards.mean().float().item()
            global_metrics[datasource]["count"] += 1

        # Calculate global averages
        logs = {}
        for datasource, metrics in global_metrics.items():
            if metrics["count"] > 0:
                if n_samples_per_prompt > 1:
                    logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
                        metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
                    )
                logs[f"eval_{datasource}_pass1"] = metrics["pass1"] / metrics["count"]

        # Log to wandb/tensorboard
        if self.wandb_logger:
            self.wandb_logger.log_eval(global_step, logs)
        if self.tensorboard_logger:
            self.tensorboard_logger.log_eval(global_step, logs)

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")
