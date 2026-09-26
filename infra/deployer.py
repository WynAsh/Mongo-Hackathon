"""DEV A. Turn an Arch document into running simulator replicas.

deploy(arch, env) -> {"short": ["http://127.0.0.1:8100", ...], "long": [...]}
teardown(env)

env names: "live0"/"live1" (blue/green, chosen by version % 2) and "shadow0".."shadow3".
Ports are deterministic, so the gateway can compute endpoints without asking anyone.
"""
from __future__ import annotations

import atexit
import subprocess
import sys
import time

import httpx

from common import config
from common.contracts import GPU_PROFILES, Arch

_procs: dict[str, list] = {}       # env -> list of Popen (fake) or container objects (docker)


def env_base_port(env: str) -> int:
    if env.startswith("live"):
        return config.LIVE_PORT_BASE + int(env[4:]) * 40
    if env.startswith("shadow"):
        return config.SHADOW_PORT_BASE + int(env[6:]) * 20
    raise ValueError(env)


def live_env_for(version: int) -> str:
    return f"live{version % 2}"


def endpoints_for(arch: Arch, env: str) -> dict[str, list[str]]:
    base = env_base_port(env)
    out = {}
    for k, name in enumerate(sorted(arch.pools)):
        out[name] = [f"http://{config.SIM_HOST}:{base + k * 8 + j}" for j in range(arch.pools[name].replicas)]
    return out


def _start_fake(port: int, prof: dict):
    return subprocess.Popen(
        [sys.executable, "-m", "infra.fakesim", "--port", str(port),
         "--max-num-seqs", str(prof["max_num_seqs"]), "--ttft-ms", str(prof["ttft_ms"]),
         "--prefill-ms-per-tok", str(prof["prefill_ms_per_tok"]), "--itl-ms", str(prof["itl_ms"]),
         "--load-factor", str(prof["load_factor"])],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _start_docker(port: int, prof: dict, name: str):
    import docker
    cli = docker.from_env()
    try:
        cli.containers.get(name).remove(force=True)
    except Exception:  # noqa: BLE001
        pass
    return cli.containers.run(
        config.SIM_IMAGE,
        command=["--model", "dummy", "--port", str(port),
                 "--max-num-seqs", str(prof["max_num_seqs"]),
                 "--time-to-first-token", f"{int(prof['ttft_ms'])}ms",
                 "--prefill-time-per-token", f"{prof['prefill_ms_per_tok']}ms",
                 "--inter-token-latency", f"{int(prof['itl_ms'])}ms",
                 "--time-factor-under-load", str(prof["load_factor"])],
        ports={f"{port}/tcp": port}, name=name, detach=True, remove=True,
    )


def wait_healthy(urls: list[str], timeout: float = 60) -> bool:
    deadline = time.time() + timeout
    pending = set(urls)
    while pending and time.time() < deadline:
        for u in list(pending):
            try:
                if httpx.get(f"{u}/v1/models", timeout=1).status_code == 200:
                    pending.discard(u)
            except Exception:  # noqa: BLE001
                pass
        if pending:
            time.sleep(0.3)
    return not pending


def deploy(arch: Arch, env: str, startup_delay: float | None = None) -> dict[str, list[str]]:
    teardown(env)
    eps = endpoints_for(arch, env)
    handles = []
    for name, urls in eps.items():
        prof = GPU_PROFILES[arch.pools[name].gpu]
        for j, url in enumerate(urls):
            port = int(url.rsplit(":", 1)[1])
            if config.SIM_MODE == "docker":
                handles.append(_start_docker(port, prof, f"ha-{env}-{name}-{j}"))
            else:
                handles.append(_start_fake(port, prof))
    _procs[env] = handles
    all_urls = [u for us in eps.values() for u in us]
    if not wait_healthy(all_urls):
        raise RuntimeError(f"replicas in {env} did not become healthy")
    time.sleep(config.REPLICA_STARTUP_S if startup_delay is None else startup_delay)  # "loading weights"
    return eps


def teardown(env: str):
    for h in _procs.pop(env, []):
        try:
            if isinstance(h, subprocess.Popen):
                h.terminate()
                h.wait(timeout=5)
            else:
                h.remove(force=True)
        except Exception:  # noqa: BLE001
            pass
    if config.SIM_MODE == "docker":  # also clean containers left by a crashed run
        try:
            import docker
            for c in docker.from_env().containers.list(all=True, filters={"name": f"ha-{env}-"}):
                c.remove(force=True)
        except Exception:  # noqa: BLE001
            pass


@atexit.register
def _cleanup():
    for env in list(_procs):
        teardown(env)
