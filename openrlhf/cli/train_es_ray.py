"""CLI entry point for Evolutionary Strategies (ES) training with Ray and vLLM.

This script trains a model using ES optimization where each vLLM engine gets a
unique parameter perturbation, and the ES gradient is computed from the scores.
"""

import argparse
import json
import os
from datetime import datetime

import ray
import torch

from openrlhf.trainer.es_trainer import ESTrainer
from openrlhf.trainer.ray import create_vllm_engines
from openrlhf.utils import get_strategy


def train(args):
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        }
        ray.init(runtime_env={"env_vars": env_vars})

    # Configure strategy for ES (minimal DeepSpeed config)
    strategy = get_strategy(args)
    strategy.print(f"ES Training with args: {args}")

    # Create vLLM engines with ES worker extension
    max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len

    vllm_engines = create_vllm_engines(
        args.vllm_num_engines,
        args.vllm_tensor_parallel_size,
        args.pretrain,
        args.seed,
        args.full_determinism,
        args.enable_prefix_caching,
        args.enforce_eager,
        max_len,
        None,  # No shared placement group for ES
        args.vllm_gpu_memory_utilization,
        args.vllm_enable_sleep,
        None,  # No logprobs_mode for ES
        agent_func_path=args.agent_func_path,
        remote_rm_url=args.remote_rm_url,
        worker_extension_cls="openrlhf.trainer.ray.es_worker_wrap.ESWorkerWrap",
        vllm_max_num_seqs=args.vllm_max_num_seqs,
    )

    # ES doesn't use actor/critic/reward/reference models - all work is done in vLLM engines
    # Pass None for these model groups
    actor_model_group = None
    critic_model_group = None
    reward_model_group = None  # ES uses rewards from agent or response directly
    reference_model_group = None

    # Create a dummy optimizer (ES optimization happens in vLLM workers)
    # This is just to satisfy the interface; LR matches --es_optimizer_params["lr"].
    dummy_params = [torch.nn.Parameter(torch.zeros(1))]
    _opt_params = json.loads(args.es_optimizer_params.strip())
    _dummy_lr = float(_opt_params.get("lr", 0.001))
    optim = torch.optim.SGD(dummy_params, lr=_dummy_lr)

    # Create ES trainer with all required parameters
    es_trainer = ESTrainer.remote(
        pretrain=args.pretrain,
        strategy=strategy,
        actor_model_group=actor_model_group,
        critic_model_group=critic_model_group,
        reward_model_group=reward_model_group,
        reference_model_group=reference_model_group,
        vllm_engines=vllm_engines,
        optim=optim,
        # Generation kwargs passed as **kwargs
        prompt_max_len=args.prompt_max_len,
        max_new_tokens=args.generate_max_len,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Run training
    ray.get(es_trainer.fit.remote())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ==================== ES-specific arguments ====================
    parser.add_argument(
        "--population_size",
        type=int,
        default=None,
        help="ES population size (number of perturbations). Defaults to vllm_num_engines.",
    )
    parser.add_argument(
        "--unique_batch_per_seed",
        action="store_true",
        default=False,
        help="Each seed gets unique batch via DistributedSampler (ZKL mode). If False, all seeds evaluate same prompts (ES mode).",
    )
    parser.add_argument("--es_std", type=float, default=0.01, help="ES noise standard deviation (sigma)")
    parser.add_argument("--es_optimizer", type=str, default="SGD", help="ES optimizer (SGD, Adam, etc.)")
    parser.add_argument(
        "--es_optimizer_params",
        type=str,
        default='{"lr": 0.001}',
        help='JSON optimizer kwargs (e.g. {"lr": 0.001}); passed to ES workers. Omitted "lr" defaults to 0.001 for the placeholder optim.',
    )
    parser.add_argument("--es_clip_grad_norm", type=float, default=0.0, help="Gradient clipping norm (0 to disable)")
    parser.add_argument("--mutate_key", type=str, default="all", help="Only mutate params containing this key")
    parser.add_argument("--mutate_exclude_key", type=str, default="none", help="Exclude params containing this key")
    parser.add_argument(
        "--auto_scale_lr_model_width",
        action="store_true",
        default=False,
        help="Auto-scale LR based on model hidden size",
    )
    parser.add_argument("--drop_center", type=float, default=0.0, help="Drop noise values below this threshold")
    parser.add_argument(
        "--es_stabilize_seed",
        action="store_true",
        default=False,
        help="Include an unperturbed seed in every batch for gradient stabilization",
    )

    # ==================== vLLM arguments ====================
    parser.add_argument(
        "--vllm_num_engines",
        type=int,
        default=4,
        help="Number of vLLM engines (parallel workers for generation)",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size per vLLM engine",
    )
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    parser.add_argument("--vllm_enable_sleep", action="store_true", default=False)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=None,
        help="vLLM max_num_seqs per engine (scheduling limit). Set to match --batch_size so a full seed batch can run in parallel.",
    )

    # ==================== Checkpoints & Logging ====================
    parser.add_argument(
        "--save_path",
        type=str,
        default="./runs/es",
        help="Run root: checkpoints/, tensorboard/, and reward metrics align with this path.",
    )
    parser.add_argument(
        "--save_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save HF checkpoints (latest on save_steps and best on eval). Use --no-save_checkpoint to disable.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=-1,
        help="Save HF checkpoint every N ES steps (optimizer updates). 0 saves once at end of training; -1 disables all HF checkpoints.",
    )
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1,
        help="Run validation every N ES steps (optimizer updates). -1 disables eval during training (step-0 baseline may still run).",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Override checkpoint directory (default: <save_path>/checkpoints).",
    )
    parser.add_argument(
        "--best_eval_metric_key",
        type=str,
        default=None,
        help="Eval metric key for best checkpoint (default: auto, prefer eval_*_pass1). Use 'none' to disable.",
    )
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # ==================== Training arguments ====================
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Maximum number of full passes over the training dataset. Default: 1.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Deprecated: use --max_epochs (same meaning).",
    )
    parser.add_argument(
        "--max_training_steps",
        type=int,
        default=None,
        help="Maximum number of ES optimizer steps (global_step). Omit for no step limit (epoch limit still applies).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of prompts per seed per ES step. Default: 64.",
    )
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=None,
        help="Deprecated: use --batch_size (prompts per seed per ES step).",
    )
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    parser.add_argument("--generate_max_len", type=int, default=1024)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--n_samples_per_prompt", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full_determinism", action="store_true", default=False)

    # ==================== Model & Data ====================
    parser.add_argument("--pretrain", type=str, required=True, help="HF model name or path")
    parser.add_argument(
        "--prompt_data", type=str, required=True, help="HF dataset name or path (comma-separated to blend)"
    )
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default=None,
        help="Sampling probabilities for blended --prompt_data (e.g. 0.5,0.5); omit to concatenate.",
    )
    parser.add_argument("--eval_dataset", type=str, default=None)
    parser.add_argument(
        "--eval_dataset_probs",
        type=str,
        default=None,
        help="Sampling probabilities for blended --eval_dataset (e.g. 0.5,0.5); omit to concatenate.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of training prompts to load from --prompt_data (truncates dataset). Default: 1000000.",
    )

    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Maximum number of eval prompts to load (defaults to --max_samples).",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input")
    parser.add_argument("--label_key", type=str, default=None)
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--remote_rm_url", type=str, default=None, help="Remote reward model URL")
    parser.add_argument("--agent_func_path", type=str, default=None, help="Agent script path for reward")

    # ==================== Logging backends ====================
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_es")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="es_%s" % datetime.now().strftime("%m%dT%H-%M"),
    )
    parser.add_argument("--use_tensorboard", type=str, default=None)
    parser.add_argument(
        "--tensorboard_text_samples",
        type=int,
        default=3,
        help="Log this many rollout examples as TensorBoard TEXT (0 disables). Train: on logging_steps; eval: every validation.",
    )
    parser.add_argument(
        "--tensorboard_text_max_chars",
        type=int,
        default=12000,
        help="Max characters per decoded sequence shown in TensorBoard text panels.",
    )

    # ==================== ModelScope parameters ====================
    parser.add_argument("--use_ms", action="store_true", default=False)

    # ==================== DeepSpeed (minimal, for strategy) ====================
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--zero_stage", type=int, default=0)  # ES doesn't use ZeRO
    parser.add_argument("--param_dtype", type=str, default="bf16")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    args = parser.parse_args()
    if args.max_samples is None:
        args.max_samples = 1000000

    if args.rollout_batch_size is not None and args.batch_size is not None:
        parser.error("Cannot specify both --rollout_batch_size and --batch_size")
    if args.rollout_batch_size is not None:
        print("[Deprecation] --rollout_batch_size is deprecated; use --batch_size")
        args.batch_size = args.rollout_batch_size
    if args.batch_size is None:
        args.batch_size = 64
    args.rollout_batch_size = args.batch_size

    if args.num_episodes is not None and args.max_epochs is not None:
        parser.error("Cannot specify both --num_episodes and --max_epochs")
    if args.num_episodes is not None:
        print("[Deprecation] --num_episodes is deprecated; use --max_epochs")
        args.max_epochs = args.num_episodes
    if args.max_epochs is None:
        args.max_epochs = 1
    if args.max_epochs < 1:
        parser.error("--max_epochs must be >= 1")
    args.num_episodes = args.max_epochs

    if args.max_training_steps is not None and args.max_training_steps < 1:
        parser.error("--max_training_steps must be a positive integer")

    optimizer_params = args.es_optimizer_params.strip()

    try:
        json.loads(optimizer_params)
    except json.JSONDecodeError:
        # Common failure: a single extra trailing brace from shell/env quoting.
        if optimizer_params.endswith("}"):
            maybe_fixed = optimizer_params[:-1].rstrip()
            try:
                json.loads(maybe_fixed)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid --es_optimizer_params JSON: {optimizer_params}") from exc
            print(
                "[Warning] Fixed invalid --es_optimizer_params by stripping a "
                "trailing '}'. Consider fixing the source config."
            )
            optimizer_params = maybe_fixed
        else:
            raise ValueError(f"Invalid --es_optimizer_params JSON: {optimizer_params}")
    args.es_optimizer_params = optimizer_params
    os.environ["ES_OPTIMIZER"] = args.es_optimizer
    os.environ["ES_OPTIMIZER_PARAMS"] = optimizer_params
    os.environ["ES_CLIP_GRAD_NORM"] = str(args.es_clip_grad_norm)
    os.environ["MUTATE_KEY"] = args.mutate_key
    os.environ["MUTATE_EXCLUDE_KEY"] = args.mutate_exclude_key
    os.environ["DROP_CENTER"] = str(args.drop_center)
    if args.auto_scale_lr_model_width:
        os.environ["AUTO_SCALE_LR_MODEL_WIDTH"] = "true"

    # Validate input_template
    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n characters instead of newline. "
            "Fix by pass newline as input_template argument in bash script."
        )

    # Default population_size to vllm_num_engines if not specified
    if args.population_size is None:
        args.population_size = args.vllm_num_engines

    # Set es_shared_batch based on unique_batch_per_seed flag
    # unique_batch_per_seed=True means es_shared_batch=False (each seed gets unique data)
    # unique_batch_per_seed=False means es_shared_batch=True (all seeds share same data)
    args.es_shared_batch = not args.unique_batch_per_seed

    # For ES with agent_func_path, set remote_rm_url to the function path
    # so SingleTurnAgentExecutor can load and use it
    if args.agent_func_path:
        args.remote_rm_url = args.agent_func_path

    args.save_path = os.path.abspath(os.path.expanduser(args.save_path))
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(args.save_path, "checkpoints")
    else:
        args.ckpt_path = os.path.abspath(os.path.expanduser(args.ckpt_path))
    if args.use_tensorboard is None:
        args.use_tensorboard = os.path.join(args.save_path, "tensorboard")
    args.use_tensorboard = os.path.abspath(os.path.expanduser(args.use_tensorboard))

    train(args)
