# serve_rm_v2.py
"""
FastAPI reward server with batching and component aggregation.

Components are loaded dynamically from a heuristics Python file.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

Values = Dict[str, List[float]]


class RewardComponent:
    """
    Return values.
    values: dict[key -> list[float]] length N
    """

    def should_call(self, label: str) -> bool:
        return True

    def __call__(self, queries: List[str], prompts: List[str], labels: List[str]) -> Values:
        raise NotImplementedError


class NamedRewardComponent:
    def __init__(self, name: str, component: RewardComponent):
        self.name = name
        self.component = component

    def should_call(self, label: str) -> bool:
        return self.component.should_call(label)

    def __call__(self, queries: List[str], prompts: List[str], labels: List[str]) -> Values:
        values = self.component(queries, prompts, labels)
        if len(values) != 1:
            raise ValueError(f"{self.name} must return exactly one reward key, got: {list(values.keys())}")
        return {self.name: next(iter(values.values()))}


class Aggregator:
    """Score samples using component routing based on should_call.

    ``components`` is a list of reward components.
    """

    def __init__(self, components: List[RewardComponent], device: torch.device):
        self.components = components
        self.device = device

    @torch.no_grad()
    def score(
        self,
        queries: List[str],
        prompts: List[str],
        labels: List[str],
    ) -> Tuple[List[float], Dict[str, Any]]:
        n = len(prompts)
        # Values are Optional[float]; None means "heuristic not evaluated for this sample"
        extra_logs: Dict[str, List[Optional[float]]] = {}

        total_rewards = torch.zeros(n, device=self.device, dtype=torch.float32)

        for comp in self.components:
            # Find indices where comp.should_call is True
            indices = [i for i, label in enumerate(labels) if comp.should_call(label)]
            if not indices:
                continue

            sub_queries = [queries[i] for i in indices]
            sub_prompts = [prompts[i] for i in indices]
            sub_labels = [labels[i] for i in indices]
            sub_n = len(indices)

            vals = comp(sub_queries, sub_prompts, sub_labels)

            for key, arr in vals.items():
                if len(arr) != sub_n:
                    raise ValueError(f"{key} len {len(arr)} != {sub_n}")

                v = torch.tensor(arr, device=self.device, dtype=torch.float32)

                # Scatter to total_rewards
                idx_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
                total_rewards.scatter_add_(0, idx_tensor, v)

                # Initialise the column with None for every sample on first encounter
                if key not in extra_logs:
                    extra_logs[key] = [None] * n

                # Scatter sub-batch results back to original positions
                for j, idx in enumerate(indices):
                    extra_logs[key][idx] = arr[j]

        rewards = total_rewards.detach().cpu().tolist()
        return rewards, extra_logs


class BatchEngine:
    def __init__(self, agg: Aggregator, batch_size: int, max_wait_s: float, queue_size: int):
        self.agg = agg
        self.batch_size = batch_size
        self.max_wait_s = max_wait_s
        self.q: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._runner())

    async def enqueue(
        self,
        queries: List[str],
        prompts: List[str],
        labels: List[str],
    ) -> Tuple[List[float], Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self.q.put((queries, prompts, labels, fut))
        return await fut

    async def _runner(self) -> None:
        while True:
            q0, p0, l0, f0 = await self.q.get()
            items = [(q0, p0, l0, f0)]
            total = len(p0)

            # Drain what is already queued
            while total < self.batch_size:
                try:
                    q1, p1, l1, f1 = self.q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                items.append((q1, p1, l1, f1))
                total += len(p1)

            # Optional tiny wait for better batching
            if self.max_wait_s > 0 and total < self.batch_size:
                t0 = asyncio.get_running_loop().time()
                while total < self.batch_size:
                    left = self.max_wait_s - (asyncio.get_running_loop().time() - t0)
                    if left <= 0:
                        break
                    try:
                        q1, p1, l1, f1 = await asyncio.wait_for(self.q.get(), timeout=left)
                    except asyncio.TimeoutError:
                        break
                    items.append((q1, p1, l1, f1))
                    total += len(p1)

            all_q: List[str] = []
            all_p: List[str] = []
            all_l: List[str] = []
            counts: List[int] = []

            for q, p, l, _ in items:
                all_q.extend(q)
                all_p.extend(p)
                all_l.extend(l)
                counts.append(len(p))

            try:
                rewards, logs = await asyncio.to_thread(self.agg.score, all_q, all_p, all_l)
            except Exception as exc:
                # Propagate error to all waiting futures so callers see
                # the exception instead of hanging forever.
                for _, _, _, fut in items:
                    if not fut.done():
                        fut.set_exception(exc)
                continue

            idx = 0
            for (_, _, _, fut), n in zip(items, counts):
                part_rewards = rewards[idx : idx + n]
                part_logs = {k: v[idx : idx + n] for k, v in logs.items()}
                idx += n
                if not fut.done():
                    fut.set_result((part_rewards, part_logs))


def create_app(
    components: List[RewardComponent],
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    max_wait_ms: float = 5.0,
    queue_size: int = 4096,
    expected_extra_log_keys: Optional[List[str]] = None,
) -> FastAPI:
    """
    Create the FastAPI reward server application.

    Args:
        components: List of reward components to apply.
        device: PyTorch device for computation
        batch_size: Batch size for reward computation
        max_wait_ms: Max wait time for batching (ms)
        queue_size: Queue size for request batching
    """
    expected_log_keys = sorted(set(expected_extra_log_keys or []))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agg = Aggregator(components, device=device)
    engine = BatchEngine(agg, batch_size=batch_size, max_wait_s=max_wait_ms / 1000.0, queue_size=queue_size)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.post("/get_reward")
    async def get_reward(request: Request) -> JSONResponse:
        try:
            data: Dict[str, Any] = await request.json()

            prompts = data.get("prompts", [])
            n = len(prompts)
            if n == 0:
                return JSONResponse({"rewards": [], "scores": [], "extra_logs": {}})

            queries = data.get("query", [])
            labels = data.get("labels", [])

            # Compute rewards
            rewards, extra_logs = await engine.enqueue(queries, prompts, labels)

            sanitized_extra_logs = {k: [v if v is not None else 0.0 for v in vals] for k, vals in extra_logs.items()}

            for key in expected_log_keys:
                if key not in sanitized_extra_logs:
                    sanitized_extra_logs[key] = [0.0] * n

            return JSONResponse({"rewards": rewards, "scores": rewards, "extra_logs": sanitized_extra_logs})
        except Exception as e:
            logger.exception("Error in /get_reward")
            return JSONResponse({"error": str(e), "rewards": [], "scores": [], "extra_logs": {}}, status_code=500)

    return app


def load_heuristics(heuristics_path: str, device: torch.device) -> Tuple[List[RewardComponent], List[str]]:
    path = Path(heuristics_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Heuristics file not found: {heuristics_path}")

    spec = importlib.util.spec_from_file_location(f"_heuristics_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load heuristics file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    heuristic_classes = getattr(module, "HEURISTICS", None)
    if not isinstance(heuristic_classes, list) or not heuristic_classes:
        raise ValueError(f"{path} must define a non-empty HEURISTICS list")

    components: List[RewardComponent] = []
    expected_log_keys: List[str] = []
    seen_names: set[str] = set()
    for cls in heuristic_classes:
        if not inspect.isclass(cls):
            raise ValueError(f"HEURISTICS entries must be classes, got: {cls!r}")
        if cls.__name__ in seen_names:
            raise ValueError(f"Duplicate heuristic class name: {cls.__name__}")
        seen_names.add(cls.__name__)

        sig = inspect.signature(cls.__init__)
        kwargs = {}
        for name, param in list(sig.parameters.items())[1:]:
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if name == "device":
                kwargs["device"] = device
                continue
            if param.default is inspect.Parameter.empty:
                raise ValueError(f"{cls.__name__} requires unsupported init arg: {name}")

        components.append(NamedRewardComponent(cls.__name__, cls(**kwargs)))
        expected_log_keys.append(cls.__name__)

    return components, expected_log_keys


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="FastAPI reward server launcher")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for reward computation")
    parser.add_argument("--max_wait_ms", type=float, default=5.0, help="Max wait time for batching (ms)")
    parser.add_argument("--heuristics", type=str, required=True, help="Path to a Python file defining HEURISTICS")
    parser.add_argument(
        "--access-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log each HTTP request to the console (default: off; use --access-log to enable).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components, expected_log_keys = load_heuristics(args.heuristics, device)

    app = create_app(
        components,
        device=device,
        batch_size=args.batch_size,
        max_wait_ms=args.max_wait_ms,
        expected_extra_log_keys=expected_log_keys,
    )

    uvicorn.run(app, host=args.host, port=args.port, access_log=args.access_log)


if __name__ == "__main__":
    main()
