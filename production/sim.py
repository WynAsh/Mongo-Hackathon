"""Run a production architecture on the simulator backends (fakesim or llm-d-inference-sim).

Each replica of a provisioned plan becomes one simulator process whose timing
flags come from a roofline estimate for that plan's GPU, model and tensor
parallelism. A persisted replay is sent at its recorded offsets. Disaggregated
prefill/decode sends each request to a prefill replica, then to a decode
replica. Prefix caching is modelled by sending only the uncached prompt
tokens. Everything runs ``TIME_SCALE`` times faster than production time and
latencies are scaled back, so results are simulator estimates, not
measurements of the real stack.
"""
import asyncio
import math
import os
import time

import httpx
from pydantic import BaseModel, Field

from common import config
from common.routing import Balancer
from infra import deployer
from production.catalog import MODELS
from traffic.generator import make_prompt

TIME_SCALE = float(os.getenv("PROD_SIM_TIME_SCALE", "0.1"))
PORT_BASE = int(os.getenv("PROD_SIM_PORT_BASE", "8400"))

# Dense BF16 TFLOPS and HBM TB/s from public datasheets (screening estimates).
GPU_SPECS = {"H200": (989, 4.8), "H100": (989, 3.35), "A100": (312, 2.0), "L40S": (362, 0.864),
             "A10G": (70, 0.6), "L4": (121, 0.3), "A10": (125, 0.6)}
MFU, BW_EFF = 0.45, 0.7
# Fraction of a request's reusable prefix actually served from cache. Modelled assumptions:
# KV-aware routers (llm-d EPP, Dynamo) keep sessions on the replica holding their cache;
# SGLang's RadixAttention also reuses branching prefixes.
CACHE_HIT = {"kserve-llmd-vllm": 0.75, "dynamo-vllm": 0.75, "dynamo-sglang": 0.92}


class ProdArch(BaseModel):
    """The architecture knobs an evolution campaign may change."""
    recipe_id: str
    model_id: str
    tensor_parallel: int = Field(ge=1)
    replicas: int = Field(ge=0)          # aggregated workers (0 when disaggregated)
    prefill_replicas: int = Field(default=0, ge=0)
    decode_replicas: int = Field(default=0, ge=0)
    max_num_seqs: int = Field(ge=1)
    max_model_len: int = Field(ge=128)
    gpu_hour_cost: float = 1.0           # 1.0 => cost is reported in GPU-hours per hour

    @classmethod
    def from_plan(cls, p, gpu_hour_cost=1.0):
        pd = p.get("disaggregation") or {}
        return cls(recipe_id=p["recipe_id"], model_id=p["model"]["model_id"],
                   tensor_parallel=p["allocation"]["tensor_parallel"],
                   replicas=0 if pd else p["allocation"]["replicas"],
                   prefill_replicas=pd.get("prefill_replicas", 0), decode_replicas=pd.get("decode_replicas", 0),
                   max_num_seqs=p["engine"]["max_num_seqs"], max_model_len=p["engine"]["max_model_len"],
                   gpu_hour_cost=gpu_hour_cost)

    @property
    def disaggregated(self):
        return self.prefill_replicas > 0

    def workers(self):
        return self.replicas + self.prefill_replicas + self.decode_replicas

    def gpus(self):
        return self.workers() * self.tensor_parallel

    def usd_hr(self):
        return round(self.gpus() * self.gpu_hour_cost, 3)

    def key(self):
        layout = f"pd{self.prefill_replicas}/{self.decode_replicas}" if self.disaggregated else f"r{self.replicas}"
        return (f"{self.recipe_id}|{self.model_id.split('/')[-1]}|tp{self.tensor_parallel}|{layout}"
                f"|seqs{self.max_num_seqs}|len{self.max_model_len}")

    def summary(self):
        layout = (f"{self.prefill_replicas} prefill + {self.decode_replicas} decode" if self.disaggregated
                  else f"{self.replicas} replicas")
        return (f"{self.recipe_id} · {self.model_id.split('/')[-1]} · {layout} × TP{self.tensor_parallel} · "
                f"seqs {self.max_num_seqs} · {self.gpus()} GPUs")


def gpu_spec(inventory):
    """(label, tflops, tbps, interconnect) for the first GPU node; unknown GPUs use A100 figures."""
    node = next((n for n in inventory["nodes"] if n.get("gpu_count")), {})
    label = str(node.get("gpu_model", "unknown"))
    for name in sorted(GPU_SPECS, key=len, reverse=True):
        if name.lower() in label.lower():
            return name, *GPU_SPECS[name], node.get("interconnect", "unknown")
    return "A100 (assumed)", *GPU_SPECS["A100"], node.get("interconnect", "unknown")


def profile(arch: ProdArch, inventory):
    """Simulator flags for one replica, in production milliseconds."""
    model = next(m for m in MODELS.values() if m["model_id"] == arch.model_id)
    _, tflops, tbps, link = gpu_spec(inventory)
    tp = arch.tensor_parallel
    tp_eff = 1.0 if tp == 1 else (0.9 if "nvlink" in str(link).lower() else 0.7)
    weights = model["parameters_b"] * 1e9 * 2
    prefill_ms_per_tok = 2 * model["parameters_b"] * 1e9 / (tflops * 1e12 * MFU * tp * tp_eff) * 1000
    itl_ms = weights / (tbps * 1e12 * BW_EFF * tp * tp_eff) * 1000
    kv_at_full = arch.max_num_seqs * model["kv_bytes_per_token"] * arch.max_model_len / 4
    load_factor = 1 + min(3.0, kv_at_full / (weights / tp))
    return {"max_num_seqs": arch.max_num_seqs, "ttft_ms": 25.0, "prefill_ms_per_tok": round(prefill_ms_per_tok, 5),
            "itl_ms": round(itl_ms, 4), "load_factor": round(load_factor, 3)}


def _scaled(prof, k):
    return {**prof, "ttft_ms": prof["ttft_ms"] * k, "prefill_ms_per_tok": prof["prefill_ms_per_tok"] * k,
            "itl_ms": prof["itl_ms"] * k}


class SimExecutor:
    """``ExperimentRunner`` executor: ``executor(arch, events, slot) -> outcomes``.

    Deployments stay warm across repeats of the same architecture; call
    ``close()`` after a stage finishes. ``prefix_reuse`` is set per traffic phase.
    """

    def __init__(self, inventory, time_scale=TIME_SCALE):
        self.inventory, self.k = inventory, time_scale
        self.prefix_reuse = 0.0
        self._warm = {}

    def _deploy(self, arch: ProdArch):
        key = arch.key()
        if key in self._warm:
            return self._warm[key]
        slot = len(self._warm)
        env = f"prod{slot}"
        deployer.teardown(env)
        prof = _scaled(profile(arch, self.inventory), self.k)
        pools = ({"prefill": arch.prefill_replicas, "decode": arch.decode_replicas} if arch.disaggregated
                 else {"shared": arch.replicas})
        base = PORT_BASE + slot * 100
        endpoints, handles = {}, []
        for i, (name, count) in enumerate(pools.items()):
            endpoints[name] = [f"http://{config.SIM_HOST}:{base + i * 48 + j}" for j in range(count)]
            for j, url in enumerate(endpoints[name]):
                port = int(url.rsplit(":", 1)[1])
                handles.append(deployer._start_docker(port, prof, f"ha-{env}-{name}-{j}") if config.SIM_MODE == "docker"
                               else deployer._start_fake(port, prof))
        deployer._procs[env] = handles
        if not deployer.wait_healthy([u for us in endpoints.values() for u in us]):
            deployer.teardown(env)
            raise RuntimeError(f"simulator replicas for {key} did not become healthy")
        self._warm[key] = (env, endpoints)
        return self._warm[key]

    def close(self):
        for env, _ in self._warm.values():
            deployer.teardown(env)
        self._warm.clear()

    def __call__(self, arch: ProdArch, events, slot=0):
        _, endpoints = self._deploy(arch)
        hit = CACHE_HIT.get(arch.recipe_id, 0.0) * self.prefix_reuse
        return asyncio.run(self._replay(arch, endpoints, events, hit))

    async def _replay(self, arch, endpoints, events, hit):
        bal, k, results = Balancer(endpoints), self.k, []
        async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=1000)) as client:
            async def call(pool, prompt, out):
                url = bal.acquire(pool)
                try:
                    r = await client.post(f"{url}/v1/chat/completions", json={
                        "model": "dummy", "max_tokens": out, "messages": [{"role": "user", "content": make_prompt(prompt)}]})
                    return None if r.status_code == 200 else f"http_{r.status_code}"
                except Exception as exc:  # noqa: BLE001
                    return type(exc).__name__
                finally:
                    bal.release(url)

            async def send(index, event):
                await asyncio.sleep(max(0, event["offset_ms"] * k / 1000 - (time.monotonic() - start)))
                prompt = max(1, math.ceil(event["prompt_tokens"] * (1 - hit)))
                t0 = time.monotonic()
                if arch.disaggregated:
                    error = await call("prefill", prompt, 1) or await call("decode", 1, event["output_tokens"])
                else:
                    error = await call("shared", prompt, event["output_tokens"])
                latency = None if error else (time.monotonic() - t0) * 1000 / k
                results.append({"event_index": index, "latency_ms": latency, "error": error})

            start = time.monotonic()
            await asyncio.gather(*(send(i, e) for i, e in enumerate(events)))
        return sorted(results, key=lambda r: r["event_index"])
