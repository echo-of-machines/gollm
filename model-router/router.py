"""
GoLLM — OpenAI-compatible API proxy with multi-backend model swap management.

Listens on port 30000 (takes over from sglang's direct host mapping).
Intercepts /v1/... requests, reads the `model` field, swaps SGLang
containers as needed via Docker SDK, then proxies the request.

Supported backends: SGLang, vLLM.

Swap strategy:
  1. Stop current model container (frees all GPU/unified memory)
  2. Start target model container (compose up if not found)
  3. Poll /health until backend is ready or timeout

In-flight drain:
  Before sleeping the current model, wait for all active proxy requests
  to complete (up to DRAIN_TIMEOUT seconds). New requests arriving during
  a pending swap queue behind the swap_lock.

Sets:
  A set is a named group of models that start together and run concurrently.
  Requests for any member model are routed to that member directly (no swap).
  Switching away from a set stops all members before loading the next model.

Architecture:
  Callers (OpenClaw, host) → GoLLM:30000 → sglang:30000 (Docker net)
  GoLLM ↔ /var/run/docker.sock (container lifecycle)
"""

import asyncio
import dataclasses
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import docker
import httpx
import yaml
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _available_ram_gb() -> float:
    """Read available RAM in GB from /proc/meminfo (MemAvailable)."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb / 1024 / 1024
    return 0.0


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("model-router")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = "/app/config.yaml"
COMPOSE_PATH = "/workbench/compose.yaml"

# Mutable config state — updated by _load_config() and hot-reload
ROUTER_CFG: dict = {}
MODELS_CFG: dict[str, dict] = {}
SETS_CFG: dict[str, dict] = {}
SGLANG_BASE_URL: str = ""
HEALTH_POLL_INTERVAL: float = 2.0
DRAIN_TIMEOUT: float = 120.0
MODEL_HOLD_SECONDS: float = 15.0
COMPOSE_PROJECT: str = ""
ALIAS_MAP: dict[str, str] = {}       # alias → model key
SET_ALIAS_MAP: dict[str, str] = {}   # alias → set key


def _load_config() -> None:
    """Load (or reload) config.yaml into module-level state."""
    global ROUTER_CFG, MODELS_CFG, SETS_CFG, SGLANG_BASE_URL, HEALTH_POLL_INTERVAL
    global DRAIN_TIMEOUT, MODEL_HOLD_SECONDS, COMPOSE_PROJECT, ALIAS_MAP, SET_ALIAS_MAP
    with open(CONFIG_PATH) as f:
        _cfg = yaml.safe_load(f)
    ROUTER_CFG = _cfg.get("router", {})
    MODELS_CFG = _cfg.get("models") or {}
    SETS_CFG = _cfg.get("sets") or {}
    SGLANG_BASE_URL = ROUTER_CFG.get("sglang_base_url", "http://sglang:30000")
    HEALTH_POLL_INTERVAL = float(ROUTER_CFG.get("health_poll_interval", 2))
    DRAIN_TIMEOUT = float(ROUTER_CFG.get("in_flight_drain_timeout", 120))
    MODEL_HOLD_SECONDS = float(ROUTER_CFG.get("model_hold_seconds", 15))
    COMPOSE_PROJECT = ROUTER_CFG.get("compose_project", "gollm")
    new_alias_map: dict[str, str] = {}
    for _key, _info in MODELS_CFG.items():
        new_alias_map[_key] = _key
        for _alias in (_info or {}).get("aliases", []):
            new_alias_map[_alias] = _key
    ALIAS_MAP = new_alias_map
    new_set_alias_map: dict[str, str] = {}
    for _skey, _sinfo in SETS_CFG.items():
        new_set_alias_map[_skey] = _skey
        for _alias in (_sinfo or {}).get("aliases", []):
            new_set_alias_map[_alias] = _skey
    SET_ALIAS_MAP = new_set_alias_map
    log.info("Config loaded: %d model(s), %d set(s) registered", len(MODELS_CFG), len(SETS_CFG))


def _save_config(cfg: dict) -> None:
    """Write config dict back to config.yaml (plain PyYAML — loses comments)."""
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _read_raw_config() -> dict:
    """Read config.yaml as a raw dict (for write operations)."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


# Initial load
_load_config()


# ---------------------------------------------------------------------------
# Backend templates — used by install to auto-generate compose services
# ---------------------------------------------------------------------------

BACKEND_TEMPLATES: dict[str, dict] = {
    "sglang": {
        "label": "SGLang",
        "image": "lmsysorg/sglang:dev-cu13",
        "port": 30000,
        "command_template": (
            "python -m sglang.launch_server"
            " --model-path {model_path}"
            " --host 0.0.0.0 --port 30000"
            " --trust-remote-code"
        ),
        "environment": [
            "HF_HOME=/root/.cache/huggingface",
        ],
        "gpu": True,
        "hf_cache": True,
        "shm_size": "16gb",
        "model_path_label": "HuggingFace model",
        "model_path_placeholder": "Qwen/Qwen3.5-35B-A3B-FP8",
    },
    "vllm": {
        "label": "vLLM",
        "image": "vllm/vllm-openai:latest",
        "port": 8000,
        "command_template": (
            "vllm serve {model_path}"
            " --host 0.0.0.0 --port 8000"
            " --trust-remote-code"
        ),
        "environment": [
            "HF_HOME=/root/.cache/huggingface",
            "VLLM_SERVER_DEV_MODE=1",
        ],
        "gpu": True,
        "hf_cache": True,
        "shm_size": "16gb",
        "model_path_label": "HuggingFace model",
        "model_path_placeholder": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    },
}


def _build_compose_service(backend: str, model_path: str) -> dict:
    """Generate a Docker Compose service dict from a backend template."""
    tpl = BACKEND_TEMPLATES[backend]
    svc: dict = {
        "profiles": ["models"],
        "image": tpl["image"],
        "expose": [str(tpl["port"])],
        "restart": "no",
    }
    if tpl.get("command_template"):
        svc["command"] = tpl["command_template"].format(model_path=model_path)
    if tpl.get("environment"):
        svc["environment"] = list(tpl["environment"])
    if tpl.get("gpu"):
        svc["deploy"] = {
            "resources": {
                "reservations": {
                    "devices": [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
                }
            }
        }
    if tpl.get("hf_cache"):
        svc["volumes"] = [
            "huggingface-cache:/root/.cache/huggingface",
            "pip-cache:/root/.cache/pip",
        ]
    if tpl.get("shm_size"):
        svc["shm_size"] = tpl["shm_size"]
    return svc


def resolve_model(model_field: str) -> Optional[str]:
    """Resolve a model alias to its config key. Returns None if unknown."""
    return ALIAS_MAP.get(model_field)


def model_base_url(model_key: Optional[str]) -> str:
    """Return the SGLang base URL for a given model key (falls back to global default)."""
    if model_key and model_key in MODELS_CFG:
        return MODELS_CFG[model_key].get("base_url", SGLANG_BASE_URL)
    return SGLANG_BASE_URL


def container_name(service: str) -> str:
    """Docker container name: {project}-{service}-1"""
    return f"{COMPOSE_PROJECT}-{service}-1"


# ---------------------------------------------------------------------------
# Async job system
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Job:
    id: str
    type: str                     # "hf-download"
    status: str                   # "pending" | "running" | "done" | "failed" | "cancelled"
    progress: float = 0.0         # 0.0–100.0
    message: str = ""
    result: Optional[dict] = None
    cancel_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
        }


JOB_REGISTRY: dict[str, Job] = {}


def _new_job(job_type: str) -> Job:
    job = Job(id=str(uuid.uuid4()), type=job_type, status="pending")
    JOB_REGISTRY[job.id] = job
    return job


async def run_hf_download(
    job: Job, repo_id: str, repo_type: str, cache_dir: str,
    on_complete=None,
) -> None:
    """Download a HuggingFace repo in a thread, reporting progress to the job."""
    job.status = "running"
    job.message = f"Downloading {repo_id}..."
    log.info("[job %s] HF download start: %s (type=%s)", job.id[:8], repo_id, repo_type)
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub import HfApi

        def _progress_callback(progress_info):  # type: ignore[no-untyped-def]
            if job.cancel_event.is_set():
                raise InterruptedError("Job cancelled")
            # progress_info is a tqdm-like object — best effort
            try:
                if hasattr(progress_info, "n") and hasattr(progress_info, "total") and progress_info.total:
                    job.progress = min(99.0, 100.0 * progress_info.n / progress_info.total)
            except Exception:
                pass

        loop = asyncio.get_event_loop()
        path = await loop.run_in_executor(
            None,
            lambda: snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                cache_dir=cache_dir,
                tqdm_class=None,   # suppress tqdm bar; we'll use size progress
            ),
        )
        if job.cancel_event.is_set():
            job.status = "cancelled"
            job.message = "Cancelled"
            return
        job.status = "done"
        job.progress = 100.0
        job.message = f"Downloaded to {path}"
        job.result = {"local_path": path}
        log.info("[job %s] HF download complete: %s", job.id[:8], path)
        if on_complete:
            try:
                on_complete()
            except Exception as e:
                log.error("[job %s] on_complete callback failed: %s", job.id[:8], e)
    except InterruptedError:
        job.status = "cancelled"
        job.message = "Cancelled"
    except Exception as e:
        job.status = "failed"
        job.message = str(e)
        log.error("[job %s] HF download failed: %s", job.id[:8], e)


# ---------------------------------------------------------------------------
# Compose write helpers (ruamel.yaml — preserves comments and block scalars)
# ---------------------------------------------------------------------------

def _ruamel() -> Optional[object]:
    """Return a ruamel.yaml YAML instance, or None if not installed."""
    try:
        from ruamel.yaml import YAML
        ry = YAML()
        ry.preserve_quotes = True
        ry.default_flow_style = False
        ry.width = 120
        return ry
    except ImportError:
        return None


def _compose_add_service(service_name: str, service_def: dict) -> None:
    """
    Add a new service block to compose.yaml, preserving existing formatting.
    Raises ValueError if service already exists.
    Raises RuntimeError if ruamel.yaml is not installed.
    """
    ry = _ruamel()
    if ry is None:
        raise RuntimeError("ruamel.yaml not installed — cannot write compose.yaml")
    import io
    with open(COMPOSE_PATH) as f:
        content = f.read()
    data = ry.load(content)
    if "services" not in data:
        data["services"] = {}
    if service_name in data["services"]:
        raise ValueError(f"Service '{service_name}' already exists in compose.yaml")
    data["services"][service_name] = service_def
    buf = io.StringIO()
    ry.dump(data, buf)
    with open(COMPOSE_PATH, "w") as f:
        f.write(buf.getvalue())
    log.info("compose.yaml: added service '%s'", service_name)


def _compose_remove_service(service_name: str) -> None:
    """
    Remove a service block from compose.yaml, preserving existing formatting.
    Raises KeyError if service not found.
    Raises RuntimeError if ruamel.yaml is not installed.
    """
    ry = _ruamel()
    if ry is None:
        raise RuntimeError("ruamel.yaml not installed — cannot write compose.yaml")
    import io
    with open(COMPOSE_PATH) as f:
        content = f.read()
    data = ry.load(content)
    services = data.get("services") or {}
    if service_name not in services:
        raise KeyError(f"Service '{service_name}' not found in compose.yaml")
    del services[service_name]
    buf = io.StringIO()
    ry.dump(data, buf)
    with open(COMPOSE_PATH, "w") as f:
        f.write(buf.getvalue())
    log.info("compose.yaml: removed service '%s'", service_name)


# ---------------------------------------------------------------------------
# State (asyncio single-threaded — no locks needed for plain assignments)
# ---------------------------------------------------------------------------

active_model: Optional[str] = None   # current config key (None when set is active)
active_set: Optional[str] = None     # current set key (None when single model is active)
swap_pending: bool = False            # a swap is queued; new requests queue too
last_request_time: float = 0.0       # monotonic time of last proxied request start
in_flight: int = 0                    # requests currently being proxied
in_flight_zero = asyncio.Event()      # set when in_flight == 0
in_flight_zero.set()

swap_lock = asyncio.Lock()            # serialises all swap operations

# Module-level httpx client (connection pooled, created in lifespan)
http_client: Optional[httpx.AsyncClient] = None
docker_client: Optional[docker.DockerClient] = None

# ---------------------------------------------------------------------------
# In-flight tracking
# ---------------------------------------------------------------------------

def _incr_in_flight() -> None:
    global in_flight, last_request_time
    in_flight += 1
    last_request_time = time.monotonic()
    in_flight_zero.clear()


def _decr_in_flight() -> None:
    global in_flight
    in_flight = max(0, in_flight - 1)
    if in_flight == 0:
        in_flight_zero.set()


async def _drain_in_flight() -> None:
    """Wait for all in-flight requests to finish before sleeping the model."""
    if in_flight == 0:
        return
    log.info("Draining %d in-flight request(s) before swap (timeout %.0fs)...", in_flight, DRAIN_TIMEOUT)
    try:
        await asyncio.wait_for(in_flight_zero.wait(), timeout=DRAIN_TIMEOUT)
        log.info("Drain complete")
    except asyncio.TimeoutError:
        log.warning("Drain timed out after %.0fs (%d still in flight) — proceeding anyway",
                    DRAIN_TIMEOUT, in_flight)

# ---------------------------------------------------------------------------
# Docker helpers (all sync SDK calls wrapped in executor)
# ---------------------------------------------------------------------------

def _container_status_sync(name: str) -> str:
    """Return container status: 'running', 'exited', 'not_found', etc."""
    try:
        c = docker_client.containers.get(name)
        c.reload()
        return c.status
    except docker.errors.NotFound:
        return "not_found"
    except Exception as e:
        log.warning("Docker status check error for %s: %s", name, e)
        return "unknown"


def _stop_container_sync(name: str) -> None:
    docker_client.containers.get(name).stop(timeout=30)


def _start_container_sync(name: str) -> None:
    docker_client.containers.get(name).start()


def _restart_container_sync(name: str) -> None:
    docker_client.containers.get(name).restart(timeout=30)


async def container_status(name: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _container_status_sync, name)


async def stop_container(name: str) -> None:
    log.info("Stopping container %s", name)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _stop_container_sync, name)
    log.info("Stopped %s", name)


async def start_container(name: str, service: str) -> None:
    """Start a stopped container. If not found, fall back to docker compose up."""
    status = await container_status(name)
    if status == "not_found":
        log.info("Container %s not found — running compose up for service %s", name, service)
        await _compose_up_service(service)
        return
    log.info("Starting container %s (current status: %s)", name, status)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _start_container_sync, name)
    log.info("Started %s", name)


async def restart_container(name: str) -> None:
    log.info("Restarting container %s", name)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _restart_container_sync, name)
    log.info("Restarted %s", name)


async def _compose_up_service(service: str) -> None:
    """Fallback: create + start a service via docker compose CLI.
    Automatically detects and passes --profile flags for profile-gated services."""
    try:
        with open(COMPOSE_PATH) as f:
            compose = yaml.safe_load(f)
        profiles = compose.get("services", {}).get(service, {}).get("profiles", [])
    except Exception:
        profiles = []

    cmd = ["docker", "compose", "-f", COMPOSE_PATH]
    for profile in profiles:
        cmd.extend(["--profile", profile])
    cmd.extend(["up", "-d", "--no-deps", service])
    log.info("Compose up: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"compose up failed (rc={proc.returncode}): {stderr.decode()[:500]}"
        )
    log.info("compose up succeeded for service %s", service)

# ---------------------------------------------------------------------------
# SGLang API calls
# ---------------------------------------------------------------------------



async def poll_health(base_url: str, timeout: float) -> bool:
    """Poll base_url/health until 200 or timeout. Returns True if healthy."""
    deadline = time.monotonic() + timeout
    log.info("Health polling %s/health (timeout=%.0fs)...", base_url, timeout)
    while time.monotonic() < deadline:
        try:
            resp = await http_client.get(f"{base_url}/health", timeout=5.0)
            if resp.status_code == 200:
                elapsed = timeout - (deadline - time.monotonic())
                log.info("SGLang healthy after %.1fs", elapsed)
                return True
        except Exception:
            pass
        await asyncio.sleep(HEALTH_POLL_INTERVAL)
    log.error("Health poll timed out after %.0fs", timeout)
    return False

# ---------------------------------------------------------------------------
# Startup: detect already-running model
# ---------------------------------------------------------------------------

async def detect_running_model() -> Optional[str]:
    """
    At startup, probe all configured service base URLs to find a running model.
    Returns the config key of the first model found healthy, or None.
    """
    checked: set[str] = set()
    for key, info in MODELS_CFG.items():
        url = info.get("base_url", SGLANG_BASE_URL)
        if url in checked:
            continue
        checked.add(url)
        try:
            resp = await http_client.get(f"{url}/v1/models", timeout=5.0)
            if resp.status_code != 200:
                continue
            data = resp.json().get("data", [])
            if not data:
                continue
            running_id = data[0].get("id", "")
            detected_key = resolve_model(running_id)
            if detected_key:
                log.info("Detected running model at %s: '%s' → config key '%s'", url, running_id, detected_key)
                return detected_key
            else:
                log.warning("Running model '%s' at %s not in config (unmanaged)", running_id, url)
        except Exception as e:
            log.info("No model at %s (likely stopped): %s", url, e)
    return None

# ---------------------------------------------------------------------------
# Core swap logic
# ---------------------------------------------------------------------------

async def _deactivate_current() -> None:
    """
    Internal: drain in-flight requests and stop the current single model.
    Must be called inside swap_lock.
    Sets active_model = None on completion.
    """
    global active_model
    if active_model is None:
        return
    current_cfg = MODELS_CFG[active_model]
    await _drain_in_flight()
    try:
        await stop_container(container_name(current_cfg["service"]))
    except Exception as e:
        log.error("Failed to stop %s: %s", active_model, e)
    active_model = None


async def _deactivate_current_set() -> None:
    """
    Internal: drain in-flight requests and stop all members of the current set.
    Must be called inside swap_lock.
    Sets active_set = None on completion.
    """
    global active_set
    if active_set is None:
        return
    members = SETS_CFG.get(active_set, {}).get("members", [])
    await _drain_in_flight()
    stop_tasks = []
    for member_key in members:
        if member_key not in MODELS_CFG:
            continue
        cname = container_name(MODELS_CFG[member_key]["service"])
        stop_tasks.append(stop_container(cname))
    if stop_tasks:
        results = await asyncio.gather(*stop_tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                log.error("Error stopping set member: %s", r)
    active_set = None


async def ensure_model(target_key: str) -> bool:
    """
    Ensure `target_key` is the active, healthy SGLang model.

    Fast path: if already active (or is a running member of the active set)
    and no swap pending, return True immediately.
    Slow path: acquire swap_lock, deactivate current model/set, activate target.

    Returns True if the model is ready to serve, False on failure.
    """
    global active_model, active_set, swap_pending

    # Fast path — target already active as single model
    if active_model == target_key and not swap_pending:
        return True

    # Fast path — target is a member of the currently active set
    if active_set is not None and not swap_pending:
        set_members = SETS_CFG.get(active_set, {}).get("members", [])
        if target_key in set_members:
            return True

    # Hold check — if the active model was used recently, wait before swapping
    if active_model is not None and active_model != target_key and MODEL_HOLD_SECONDS > 0:
        hold_remaining = MODEL_HOLD_SECONDS - (time.monotonic() - last_request_time)
        if hold_remaining > 0:
            log.info(
                "Hold: '%s' used %.1fs ago — waiting %.1fs before swap to '%s'",
                active_model, time.monotonic() - last_request_time, hold_remaining, target_key,
            )
            await asyncio.sleep(hold_remaining)
            if active_model == target_key and not swap_pending:
                return True

    swap_pending = True

    async with swap_lock:
        # Re-check inside lock
        if active_model == target_key:
            swap_pending = False
            return True
        if active_set is not None:
            set_members = SETS_CFG.get(active_set, {}).get("members", [])
            if target_key in set_members:
                swap_pending = False
                return True

        log.info("─── Swap: %s → %s ───", active_model or f"set:{active_set}" or "none", target_key)

        target_cfg = MODELS_CFG[target_key]
        target_service = target_cfg["service"]
        target_container = container_name(target_service)
        target_url = model_base_url(target_key)
        start_timeout = float(target_cfg.get("health_timeout_start", 300))

        # ── Step 1: Deactivate current model or set ───────────────────────
        await _deactivate_current()      # no-op if active_model is None
        await _deactivate_current_set()  # no-op if active_set is None

        # ── Step 1b: RAM check ─────────────────────────────────────────────
        ram_required_gb = float(target_cfg.get("ram_required_gb", 0))
        if ram_required_gb > 0:
            available_gb = _available_ram_gb()
            if available_gb < ram_required_gb:
                log.error(
                    "Insufficient RAM to load '%s': need %.1f GB, only %.1f GB available",
                    target_key, ram_required_gb, available_gb,
                )
                swap_pending = False
                return False
            log.info("RAM check OK: need %.1f GB, %.1f GB available", ram_required_gb, available_gb)

        # ── Step 2: Activate target model ─────────────────────────────────
        status = await container_status(target_container)
        log.info("Container %s status: %s", target_container, status)

        healthy = False

        if status == "running":
            healthy = await poll_health(target_url, start_timeout)
            if not healthy:
                log.info("Running but unhealthy — restarting %s", target_container)
                try:
                    await restart_container(target_container)
                    healthy = await poll_health(target_url, start_timeout)
                except Exception as e:
                    log.error("Restart failed for %s: %s", target_container, e)

        elif status in ("exited", "created", "paused", "not_found"):
            try:
                await start_container(target_container, target_service)
                healthy = await poll_health(target_url, start_timeout)
            except Exception as e:
                log.error("Failed to start %s: %s", target_container, e)

        else:
            log.error("Unexpected container status '%s' for %s", status, target_container)

        # ── Step 3: Update state ───────────────────────────────────────────
        if healthy:
            active_model = target_key
            log.info("─── Swap complete. Active: %s ───", target_key)
        else:
            log.error("Swap to %s FAILED — health check did not pass", target_key)

        swap_pending = False
        return healthy


async def ensure_set(target_set_key: str) -> bool:
    """
    Ensure all members of `target_set_key` are active and healthy.

    Fast path: set already active and no swap pending → True immediately.
    Slow path: stop current model/set, start all members in parallel.

    Returns True if all members are healthy, False on failure.
    """
    global active_model, active_set, swap_pending

    # Fast path
    if active_set == target_set_key and not swap_pending:
        return True

    swap_pending = True

    async with swap_lock:
        if active_set == target_set_key:
            swap_pending = False
            return True

        set_cfg = SETS_CFG[target_set_key]
        members: list[str] = set_cfg.get("members", [])
        if not members:
            log.error("Set '%s' has no members", target_set_key)
            swap_pending = False
            return False

        log.info("─── Set swap → %s (%d members) ───", target_set_key, len(members))

        # ── RAM check: sum of all members ─────────────────────────────────
        total_ram = sum(
            float(MODELS_CFG.get(m, {}).get("ram_required_gb", 0)) for m in members
        )
        if total_ram > 0:
            available_gb = _available_ram_gb()
            if available_gb < total_ram:
                log.error(
                    "Insufficient RAM for set '%s': need %.1f GB, only %.1f GB available",
                    target_set_key, total_ram, available_gb,
                )
                swap_pending = False
                return False
            log.info("RAM check OK for set '%s': need %.1f GB, %.1f GB available",
                     target_set_key, total_ram, available_gb)

        # ── Deactivate current state ───────────────────────────────────────
        await _deactivate_current()
        await _deactivate_current_set()

        # ── Start all members in parallel ─────────────────────────────────
        async def _start_member(member_key: str) -> bool:
            if member_key not in MODELS_CFG:
                log.error("Set member '%s' not in models config", member_key)
                return False
            info = MODELS_CFG[member_key]
            cname = container_name(info["service"])
            try:
                await start_container(cname, info["service"])
                start_timeout = float(info.get("health_timeout_start", 300))
                return await poll_health(model_base_url(member_key), start_timeout)
            except Exception as e:
                log.error("Failed to start set member '%s': %s", member_key, e)
                return False

        results = await asyncio.gather(*[_start_member(m) for m in members])
        all_healthy = all(results)

        if all_healthy:
            active_set = target_set_key
            active_model = None
            log.info("─── Set '%s' active (%d models) ───", target_set_key, len(members))
        else:
            failed = [members[i] for i, ok in enumerate(results) if not ok]
            log.error("Set '%s' startup failed — unhealthy members: %s", target_set_key, failed)

        swap_pending = False
        return all_healthy

# ---------------------------------------------------------------------------
# Routing helper
# ---------------------------------------------------------------------------

async def resolve_and_ensure(model_field: str) -> tuple[bool, Optional[str]]:
    """
    Given a model field from a request body, ensure the right model/set is active.
    Returns (ready: bool, base_url: str | None).

    Priority:
      1. Direct model alias → load single model
         Fast path if it's the active single model, or a member of the active set.
      2. Set alias → load all set members
         Fast path if the set is already active.
      3. Unknown → route to whatever is currently active.
    """
    if model_field:
        # Check direct model alias
        model_key = ALIAS_MAP.get(model_field)
        if model_key:
            # Fast path: single model active
            if active_model == model_key and not swap_pending:
                return True, model_base_url(model_key)
            # Fast path: model is a member of the active set
            if active_set is not None and not swap_pending:
                if model_key in SETS_CFG.get(active_set, {}).get("members", []):
                    return True, model_base_url(model_key)
            # Need to load
            ready = await ensure_model(model_key)
            return ready, model_base_url(model_key)

        # Check set alias
        set_key = SET_ALIAS_MAP.get(model_field)
        if set_key:
            if active_set == set_key and not swap_pending:
                members = SETS_CFG[set_key].get("members", [])
                primary = SETS_CFG[set_key].get("primary") or (members[0] if members else None)
                return True, model_base_url(primary) if primary else None
            ready = await ensure_set(set_key)
            if ready:
                members = SETS_CFG[set_key].get("members", [])
                primary = SETS_CFG[set_key].get("primary") or (members[0] if members else None)
                return ready, model_base_url(primary) if primary else None
            return False, None

        # Unknown model field — log and fall through to active model
        log.warning("Unknown model '%s' in request — routing to currently active model", model_field)

    # No model field or unknown — use whatever is active
    if active_model:
        return True, model_base_url(active_model)
    if active_set:
        members = SETS_CFG.get(active_set, {}).get("members", [])
        primary = SETS_CFG[active_set].get("primary") or (members[0] if members else None)
        return True, model_base_url(primary) if primary else None

    return False, None

# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------

# Headers that must not be forwarded (hop-by-hop)
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
})


async def proxy_request(
    method: str,
    path: str,
    headers: dict,
    body: bytes,
    params: dict,
    base_url: Optional[str] = None,
) -> Response:
    """
    Forward a request to the active SGLang instance.
    Tracks in-flight count; decrements on completion or error.
    Handles both streaming (SSE) and buffered responses.
    If base_url is not provided, falls back to model_base_url(active_model).
    """
    if base_url is None:
        base_url = model_base_url(active_model)
    url = f"{base_url}{path}"

    forward_headers = {
        k: v for k, v in headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
    }
    forward_headers["host"] = base_url.split("//", 1)[-1]

    _incr_in_flight()
    try:
        req = http_client.build_request(
            method=method,
            url=url,
            headers=forward_headers,
            content=body,
            params=params,
        )
        resp = await http_client.send(req, stream=True)

        content_type = resp.headers.get("content-type", "")
        is_streaming = "text/event-stream" in content_type

        resp_headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() not in _HOP_BY_HOP
        }
        resp_headers["x-model-router"] = active_model or (f"set:{active_set}" if active_set else "unknown")

        if is_streaming:
            async def stream_gen() -> AsyncIterator[bytes]:
                try:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                finally:
                    await resp.aclose()
                    _decr_in_flight()

            return StreamingResponse(
                stream_gen(),
                status_code=resp.status_code,
                headers=resp_headers,
                media_type=content_type,
            )
        else:
            try:
                content = await resp.aread()
            finally:
                await resp.aclose()
            _decr_in_flight()
            return Response(
                content=content,
                status_code=resp.status_code,
                headers=resp_headers,
                media_type=content_type or "application/json",
            )

    except Exception as e:
        _decr_in_flight()
        log.error("Proxy error %s %s: %s", method, path, e)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Upstream proxy error: {e}", "type": "proxy_error"}},
        )

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, docker_client, active_model

    log.info("GoLLM starting")

    http_client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0))

    try:
        docker_client = docker.from_env()
        log.info("Docker SDK connected")
    except Exception as e:
        log.error("Docker SDK connection failed: %s", e)

    active_model = await detect_running_model()

    default_key = ROUTER_CFG.get("default_model")
    if active_model is None and default_key and default_key in MODELS_CFG:
        log.info("No model detected at startup — pre-warming default model '%s'", default_key)
        ready = await ensure_model(default_key)
        if ready:
            log.info("Pre-warm complete — '%s' is healthy and ready", default_key)
        else:
            log.warning("Pre-warm failed for '%s' — router is up but model not active", default_key)

    log.info("Startup complete — active model: %s", active_model or "none")

    yield

    log.info("GoLLM shutting down")
    await http_client.aclose()
    if docker_client:
        docker_client.close()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="GoLLM", version="2.0.0", lifespan=lifespan)

# Serve GUI static files if the directory exists
_static_dir = "/app/static"
if os.path.isdir(_static_dir) and os.listdir(_static_dir):
    app.mount("/ui", StaticFiles(directory=_static_dir, html=True), name="ui")
    log.info("GUI available at /ui")

# ---------------------------------------------------------------------------
# Management endpoints
# ---------------------------------------------------------------------------

@app.get("/favicon.ico")
async def favicon():
    """Return empty 204 to stop favicon requests hitting the proxy."""
    return Response(status_code=204)


@app.get("/health")
async def health():
    """Router liveness — always 200 if the router process is alive."""
    return {"status": "ok", "active_model": active_model, "active_set": active_set}


@app.get("/status")
async def status():
    """Detailed state: active model/set, in-flight count, all configured models."""
    return {
        "active_model": active_model,
        "active_set": active_set,
        "swap_pending": swap_pending,
        "in_flight": in_flight,
        "models": {
            key: {
                "service": info["service"],
                "model_path": info["model_path"],
                "aliases": info.get("aliases", []),
                "active": key == active_model,
                "in_active_set": (
                    active_set is not None and
                    key in SETS_CFG.get(active_set, {}).get("members", [])
                ),
            }
            for key, info in MODELS_CFG.items()
        },
    }


@app.get("/router/system")
async def system_info():
    """System resource info: RAM, active model/set, job counts."""
    available_gb = _available_ram_gb()
    total_ram_kb = 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total_ram_kb = int(line.split()[1])
                break
    total_gb = total_ram_kb / 1024 / 1024
    job_counts = {}
    for j in JOB_REGISTRY.values():
        job_counts[j.status] = job_counts.get(j.status, 0) + 1
    return {
        "ram_total_gb": round(total_gb, 1),
        "ram_available_gb": round(available_gb, 1),
        "ram_used_gb": round(total_gb - available_gb, 1),
        "active_model": active_model,
        "active_set": active_set,
        "swap_pending": swap_pending,
        "in_flight": in_flight,
        "jobs": job_counts,
        "model_count": len(MODELS_CFG),
        "set_count": len(SETS_CFG),
    }


@app.post("/router/swap/{model_key}")
async def manual_swap(model_key: str):
    """Manually trigger a model swap without sending an inference request."""
    canonical = resolve_model(model_key)
    if canonical is None:
        # Try set
        set_key = SET_ALIAS_MAP.get(model_key)
        if set_key:
            success = await ensure_set(set_key)
            if success:
                return {"status": "ok", "active_set": active_set}
            return JSONResponse(status_code=503, content={"error": f"Set swap to '{set_key}' failed"})
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown model/set '{model_key}'. "
                     f"Available models: {list(MODELS_CFG.keys())}, sets: {list(SETS_CFG.keys())}"},
        )
    success = await ensure_model(canonical)
    if success:
        return {"status": "ok", "active_model": active_model}
    return JSONResponse(
        status_code=503,
        content={"error": f"Swap to '{canonical}' failed — check router logs"},
    )


@app.get("/v1/models")
async def list_models():
    """Return all configured models. OpenAI clients use this to validate model names."""
    return {
        "object": "list",
        "data": [
            {
                "id": info["model_path"],
                "object": "model",
                "owned_by": "gollm",
                "active": key == active_model,
                "aliases": info.get("aliases", []),
            }
            for key, info in MODELS_CFG.items()
        ],
    }


# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------

@app.post("/router/config/reload")
async def config_reload():
    """Hot-reload config.yaml without restarting the router."""
    _load_config()
    return {"status": "ok", "models": len(MODELS_CFG), "sets": len(SETS_CFG)}


@app.get("/router/backends")
async def list_backends():
    """Return available inference backends for the install dialog."""
    backends = []
    for key, tpl in BACKEND_TEMPLATES.items():
        backends.append({
            "key": key,
            "label": tpl["label"],
            "port": tpl["port"],
            "model_path_label": tpl.get("model_path_label", "Model path"),
            "model_path_placeholder": tpl.get("model_path_placeholder", ""),
        })
    return {"backends": backends}


@app.get("/router/hf-token")
async def get_hf_token():
    """Check if an HF token is configured (returns masked value, never the full token)."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return {"configured": True, "masked": token[:5] + "..." + token[-4:]}
    return {"configured": False, "masked": ""}


@app.post("/router/hf-token")
async def set_hf_token(request: Request):
    """Set the HF_TOKEN environment variable at runtime."""
    body = await request.json()
    token = body.get("token", "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        return {"status": "ok", "masked": token[:5] + "..." + token[-4:]}
    return JSONResponse(status_code=400, content={"error": "Token is required"})


@app.delete("/router/hf-token")
async def remove_hf_token():
    """Remove the HF_TOKEN environment variable."""
    os.environ.pop("HF_TOKEN", None)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Model management — register / unregister / install / uninstall / status
# ---------------------------------------------------------------------------

@app.get("/router/models")
async def router_list_models():
    """List all registered models with container status."""
    result = []
    for key, info in MODELS_CFG.items():
        cname = container_name(info["service"])
        cstatus = await container_status(cname)
        result.append({
            "key": key,
            "service": info["service"],
            "model_path": info.get("model_path", ""),
            "aliases": info.get("aliases", []),
            "ram_required_gb": info.get("ram_required_gb", 0),
            "active": key == active_model,
            "in_active_set": (
                active_set is not None and
                key in SETS_CFG.get(active_set, {}).get("members", [])
            ),
            "container_status": cstatus,
        })
    return {"models": result}


@app.post("/router/models")
async def register_model(request: Request):
    """Register a new model in config.yaml. Body: {key, service, model_path, base_url, aliases, ...}"""
    body = await request.json()
    key = body.get("key")
    if not key:
        return JSONResponse(status_code=400, content={"error": "key is required"})
    if key in MODELS_CFG:
        return JSONResponse(status_code=409, content={"error": f"Model '{key}' already registered"})
    cfg = _read_raw_config()
    cfg.setdefault("models", {})[key] = {k: v for k, v in body.items() if k != "key"}
    _save_config(cfg)
    _load_config()
    return {"status": "ok", "key": key}


@app.post("/router/models/install")
async def install_model(request: Request):
    """
    Install a new model: write service to compose.yaml, register in config.yaml,
    and optionally start an async HF download job.

    Body:
    {
      "key": "my-model",
      "backend": "sglang" | "vllm",          // selects template
      "model_path": "org/model-name",
      "service": "sglang-mymodel",            // optional, auto-derived from backend+key
      "base_url": "http://sglang-mymodel:30000",  // optional, auto-derived
      "ram_required_gb": 30,
      "aliases": [...],
      "health_timeout_start": 300,
      "compose_service": { ... },             // optional, overrides template
      "download": true
    }
    """
    body = await request.json()
    key = body.get("key")
    if not key:
        return JSONResponse(status_code=400, content={"error": "key is required"})
    if key in MODELS_CFG:
        return JSONResponse(status_code=409, content={"error": f"Model '{key}' already registered"})

    backend = body.get("backend", "sglang")
    model_path = body.get("model_path", "")
    service_name = body.get("service", f"{backend}-{key}")
    compose_service_def = body.get("compose_service")

    # Auto-derive base_url from backend template if not provided
    if not body.get("base_url"):
        tpl = BACKEND_TEMPLATES.get(backend)
        port = tpl["port"] if tpl else 30000
        body["base_url"] = f"http://{service_name}:{port}"

    # Auto-generate compose service from backend template if not explicitly provided
    if not compose_service_def and backend in BACKEND_TEMPLATES and model_path:
        compose_service_def = _build_compose_service(backend, model_path)

    # Build the registration closure (compose + config writes)
    model_entry = {k: v for k, v in body.items()
                   if k not in ("key", "compose_service", "download")}

    def _register_model():
        """Write compose service + config entry. Called immediately or after download."""
        compose_ok = False
        if compose_service_def:
            try:
                _compose_add_service(service_name, compose_service_def)
                compose_ok = True
            except (ValueError, RuntimeError) as e:
                log.error("Failed to write compose service '%s': %s", service_name, e)
        cfg = _read_raw_config()
        cfg.setdefault("models", {})[key] = model_entry
        _save_config(cfg)
        _load_config()
        log.info("Model '%s' registered (compose=%s)", key, compose_ok)

    # If download requested, defer registration until download completes
    job_id = None
    if body.get("download") and model_path:
        job = _new_job("hf-download")
        job_id = job.id
        asyncio.create_task(
            run_hf_download(
                job=job,
                repo_id=model_path,
                repo_type="model",
                cache_dir="/root/.cache/huggingface",
                on_complete=_register_model,
            )
        )
        return {
            "status": "ok",
            "key": key,
            "compose_written": False,
            "download_job_id": job_id,
            "deferred": True,
        }

    # No download — register immediately
    _register_model()
    return {
        "status": "ok",
        "key": key,
        "compose_written": bool(compose_service_def),
        "download_job_id": None,
    }


@app.delete("/router/models/{key}")
async def unregister_model(key: str):
    """Unregister a model from config.yaml. Model must not be active."""
    canonical = resolve_model(key)
    if not canonical:
        return JSONResponse(status_code=404, content={"error": f"Unknown model '{key}'"})
    if canonical == active_model:
        return JSONResponse(status_code=409, content={"error": f"Cannot unregister active model '{canonical}'. Stop it first."})
    cfg = _read_raw_config()
    cfg.get("models", {}).pop(canonical, None)
    _save_config(cfg)
    _load_config()
    return {"status": "ok", "key": canonical}


@app.delete("/router/models/{key}/uninstall")
async def uninstall_model(key: str, remove_service: bool = False):
    """
    Uninstall a model: unregister from config.yaml and optionally remove from compose.yaml.
    Model must not be active.
    Query param: ?remove_service=true to also remove the compose service.
    """
    canonical = resolve_model(key)
    if not canonical:
        return JSONResponse(status_code=404, content={"error": f"Unknown model '{key}'"})
    if canonical == active_model:
        return JSONResponse(status_code=409, content={"error": f"Cannot uninstall active model '{canonical}'. Stop it first."})
    info = MODELS_CFG[canonical]
    service_name = info["service"]

    compose_removed = False
    if remove_service:
        try:
            _compose_remove_service(service_name)
            compose_removed = True
        except (KeyError, RuntimeError) as e:
            log.warning("compose remove service failed: %s", e)

    cfg = _read_raw_config()
    cfg.get("models", {}).pop(canonical, None)
    _save_config(cfg)
    _load_config()

    return {"status": "ok", "key": canonical, "compose_removed": compose_removed}


@app.get("/router/models/{key}/status")
async def model_status_detail(key: str):
    """Get container status and health for a specific model."""
    canonical = resolve_model(key)
    if not canonical:
        return JSONResponse(status_code=404, content={"error": f"Unknown model '{key}'"})
    info = MODELS_CFG[canonical]
    cname = container_name(info["service"])
    cstatus = await container_status(cname)
    healthy = False
    if cstatus == "running":
        try:
            resp = await http_client.get(f"{model_base_url(canonical)}/health", timeout=3.0)
            healthy = resp.status_code == 200
        except Exception:
            pass
    return {
        "key": canonical,
        "active": canonical == active_model,
        "in_active_set": (
            active_set is not None and
            canonical in SETS_CFG.get(active_set, {}).get("members", [])
        ),
        "container_status": cstatus,
        "healthy": healthy,
        "service": info["service"],
        "container_name": cname,
        "model_path": info.get("model_path", ""),
    }


# ---------------------------------------------------------------------------
# Model lifecycle — start / stop / sleep / wake
# ---------------------------------------------------------------------------

@app.post("/router/models/{key}/start")
async def model_start(key: str):
    """Cold-start a model container and wait for it to be healthy."""
    canonical = resolve_model(key)
    if not canonical:
        return JSONResponse(status_code=404, content={"error": f"Unknown model '{key}'"})
    info = MODELS_CFG[canonical]
    ram_required = float(info.get("ram_required_gb", 0))
    if ram_required > 0:
        available = _available_ram_gb()
        if available < ram_required:
            return JSONResponse(status_code=507, content={
                "error": f"Insufficient RAM: need {ram_required}GB, {available:.1f}GB available"
            })
    cname = container_name(info["service"])

    try:
        await start_container(cname, info["service"])
        start_timeout = float(info.get("health_timeout_start", 300))
        healthy = await poll_health(model_base_url(canonical), start_timeout)
        if healthy:
            return {"status": "ok", "key": canonical, "container": cname}
        return JSONResponse(status_code=503, content={"error": "Container started but health check timed out"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/router/models/{key}/stop")
async def model_stop(key: str):
    """Stop a model container, draining in-flight requests if it is active."""
    global active_model
    canonical = resolve_model(key)
    if not canonical:
        return JSONResponse(status_code=404, content={"error": f"Unknown model '{key}'"})
    info = MODELS_CFG[canonical]
    cname = container_name(info["service"])
    if canonical == active_model:
        await _drain_in_flight()
        active_model = None

    try:
        await stop_container(cname)
        return {"status": "ok", "key": canonical}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



# ---------------------------------------------------------------------------
# Async jobs
# ---------------------------------------------------------------------------

@app.get("/router/jobs")
async def list_jobs():
    """List all jobs (HF downloads, etc.)."""
    return {"jobs": [j.to_dict() for j in JOB_REGISTRY.values()]}


@app.post("/router/jobs/hf-download")
async def start_hf_download(request: Request):
    """
    Start an async HuggingFace model download.
    Body: {"repo_id": "org/model", "repo_type": "model", "cache_dir": "/root/.cache/huggingface"}
    """
    body = await request.json()
    repo_id = body.get("repo_id")
    if not repo_id:
        return JSONResponse(status_code=400, content={"error": "repo_id is required"})
    repo_type = body.get("repo_type", "model")
    cache_dir = body.get("cache_dir", "/root/.cache/huggingface")
    job = _new_job("hf-download")
    asyncio.create_task(run_hf_download(job, repo_id, repo_type, cache_dir))
    return {"status": "ok", "job_id": job.id, "repo_id": repo_id}


@app.get("/router/jobs/{job_id}")
async def get_job(job_id: str):
    """Get status of a specific job."""
    job = JOB_REGISTRY.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": f"Job '{job_id}' not found"})
    return job.to_dict()


@app.post("/router/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Request cancellation of a running job."""
    job = JOB_REGISTRY.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": f"Job '{job_id}' not found"})
    if job.status not in ("pending", "running"):
        return JSONResponse(status_code=409, content={"error": f"Job is already '{job.status}'"})
    job.cancel_event.set()
    return {"status": "ok", "job_id": job_id, "message": "Cancellation requested"}


@app.delete("/router/jobs/{job_id}")
async def delete_job(job_id: str):
    """Remove a completed/failed/cancelled job from the registry."""
    job = JOB_REGISTRY.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": f"Job '{job_id}' not found"})
    if job.status in ("pending", "running"):
        return JSONResponse(status_code=409, content={"error": "Cannot delete a running job. Cancel it first."})
    del JOB_REGISTRY[job_id]
    return {"status": "ok", "job_id": job_id}


# ---------------------------------------------------------------------------
# Set management
# ---------------------------------------------------------------------------

@app.get("/router/sets")
async def list_sets():
    """List all configured sets with member status."""
    result = []
    for set_key, set_info in SETS_CFG.items():
        members = set_info.get("members", [])
        member_statuses = []
        for m in members:
            if m in MODELS_CFG:
                cname = container_name(MODELS_CFG[m]["service"])
                cstatus = await container_status(cname)
            else:
                cstatus = "not_configured"
            member_statuses.append({"key": m, "container_status": cstatus})
        result.append({
            "key": set_key,
            "description": set_info.get("description", ""),
            "aliases": set_info.get("aliases", []),
            "primary": set_info.get("primary"),
            "members": member_statuses,
            "ram_required_gb": sum(
                float(MODELS_CFG.get(m, {}).get("ram_required_gb", 0)) for m in members
            ),
            "active": set_key == active_set,
        })
    return {"sets": result}


@app.post("/router/sets")
async def create_set(request: Request):
    """
    Create a new model set in config.yaml.
    Body: {key, members: [model_key, ...], aliases: [...], primary: model_key, description: ""}
    """
    body = await request.json()
    key = body.get("key")
    if not key:
        return JSONResponse(status_code=400, content={"error": "key is required"})
    if key in SETS_CFG:
        return JSONResponse(status_code=409, content={"error": f"Set '{key}' already exists"})
    members = body.get("members", [])
    if not members:
        return JSONResponse(status_code=400, content={"error": "members list is required"})
    # Validate members
    unknown = [m for m in members if m not in MODELS_CFG]
    if unknown:
        return JSONResponse(status_code=400, content={"error": f"Unknown model keys: {unknown}"})
    cfg = _read_raw_config()
    cfg.setdefault("sets", {})[key] = {k: v for k, v in body.items() if k != "key"}
    _save_config(cfg)
    _load_config()
    return {"status": "ok", "key": key}


@app.delete("/router/sets/{key}")
async def delete_set(key: str):
    """Delete a set from config.yaml. Set must not be active."""
    set_key = SET_ALIAS_MAP.get(key, key)
    if set_key not in SETS_CFG:
        return JSONResponse(status_code=404, content={"error": f"Set '{key}' not found"})
    if set_key == active_set:
        return JSONResponse(status_code=409, content={"error": f"Cannot delete active set '{set_key}'. Stop it first."})
    cfg = _read_raw_config()
    cfg.get("sets", {}).pop(set_key, None)
    _save_config(cfg)
    _load_config()
    return {"status": "ok", "key": set_key}


@app.get("/router/sets/{key}/status")
async def set_status(key: str):
    """Get detailed status for a set and all its members."""
    set_key = SET_ALIAS_MAP.get(key, key)
    if set_key not in SETS_CFG:
        return JSONResponse(status_code=404, content={"error": f"Set '{key}' not found"})
    set_info = SETS_CFG[set_key]
    members = set_info.get("members", [])
    member_details = []
    for m in members:
        if m not in MODELS_CFG:
            member_details.append({"key": m, "error": "not in models config"})
            continue
        cname = container_name(MODELS_CFG[m]["service"])
        cstatus = await container_status(cname)
        healthy = False
        if cstatus == "running":
            try:
                resp = await http_client.get(f"{model_base_url(m)}/health", timeout=3.0)
                healthy = resp.status_code == 200
            except Exception:
                pass
        member_details.append({
            "key": m,
            "container_status": cstatus,
            "healthy": healthy,
            "service": MODELS_CFG[m]["service"],
        })
    return {
        "key": set_key,
        "active": set_key == active_set,
        "description": set_info.get("description", ""),
        "members": member_details,
    }


@app.post("/router/sets/{key}/start")
async def set_start(key: str):
    """Load all members of a set (parallel start)."""
    set_key = SET_ALIAS_MAP.get(key, key)
    if set_key not in SETS_CFG:
        return JSONResponse(status_code=404, content={"error": f"Set '{key}' not found"})
    success = await ensure_set(set_key)
    if success:
        return {"status": "ok", "active_set": active_set}
    return JSONResponse(status_code=503, content={"error": f"Set '{set_key}' startup failed — check logs"})


@app.post("/router/sets/{key}/stop")
async def set_stop(key: str):
    """Stop all members of a set."""
    global active_set
    set_key = SET_ALIAS_MAP.get(key, key)
    if set_key not in SETS_CFG:
        return JSONResponse(status_code=404, content={"error": f"Set '{key}' not found"})
    members = SETS_CFG[set_key].get("members", [])
    if set_key == active_set:
        await _drain_in_flight()
    errors = []
    for m in members:
        if m not in MODELS_CFG:
            continue
        cname = container_name(MODELS_CFG[m]["service"])
        try:
            await stop_container(cname)
        except Exception as e:
            errors.append({"member": m, "error": str(e)})
    if set_key == active_set:
        active_set = None
    if errors:
        return JSONResponse(status_code=207, content={"status": "partial", "errors": errors})
    return {"status": "ok", "key": set_key}


@app.put("/router/sets/{key}")
async def update_set(key: str, request: Request):
    """Update set configuration (members, aliases, primary, description)."""
    set_key = SET_ALIAS_MAP.get(key, key)
    if set_key not in SETS_CFG:
        return JSONResponse(status_code=404, content={"error": f"Set '{key}' not found"})
    body = await request.json()
    if "members" in body:
        unknown = [m for m in body["members"] if m not in MODELS_CFG]
        if unknown:
            return JSONResponse(status_code=400, content={"error": f"Unknown model keys: {unknown}"})
    cfg = _read_raw_config()
    existing = cfg.get("sets", {}).get(set_key, {})
    existing.update({k: v for k, v in body.items() if k != "key"})
    cfg.setdefault("sets", {})[set_key] = existing
    _save_config(cfg)
    _load_config()
    return {"status": "ok", "key": set_key}


# ---------------------------------------------------------------------------
# Compose service controls
# ---------------------------------------------------------------------------

@app.get("/router/services")
async def list_services():
    """List all services defined in compose.yaml with their container status."""
    try:
        with open(COMPOSE_PATH) as f:
            compose = yaml.safe_load(f)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Cannot read compose.yaml: {e}"})
    result = []
    for svc_name, svc_def in (compose.get("services") or {}).items():
        if not svc_def:
            continue
        cname = container_name(svc_name)
        cstatus = await container_status(cname)
        result.append({
            "service": svc_name,
            "container": cname,
            "status": cstatus,
            "profiles": (svc_def or {}).get("profiles", []),
            "image": (svc_def or {}).get("image", "<build>"),
        })
    return {"services": result}


@app.post("/router/services/{service}/start")
async def service_start(service: str):
    """Start a compose service (docker compose up --no-deps)."""
    try:
        await _compose_up_service(service)
        return {"status": "ok", "service": service}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/router/services/{service}/stop")
async def service_stop(service: str):
    """Stop a compose service container."""
    cname = container_name(service)
    cstatus = await container_status(cname)
    if cstatus == "not_found":
        return JSONResponse(status_code=404, content={"error": f"Container '{cname}' not found"})
    try:
        await stop_container(cname)
        return {"status": "ok", "service": service}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/router/services/{service}/restart")
async def service_restart(service: str):
    """Restart a compose service container."""
    cname = container_name(service)
    try:
        await restart_container(cname)
        return {"status": "ok", "service": service}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Main proxy: all /v1/... routes (except /v1/models handled above)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# WebSocket — live container log stream
# ---------------------------------------------------------------------------

async def _docker_log_generator(ctr_name: str):
    """Yield log lines from a Docker container via subprocess (non-blocking)."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "logs", "-f", "--tail", "80", "--timestamps", ctr_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield line.decode("utf-8", errors="replace")
    except asyncio.CancelledError:
        proc.kill()
        raise
    except Exception as e:
        yield f"[GoLLM] Log stream error: {e}\n"
    finally:
        if proc.returncode is None:
            proc.kill()


@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket):
    """
    Stream Docker container logs to the browser.
    Client sends: {"container": "gollm-sglang-mymodel-1"} to switch streams.
    Server sends: {"type": "log", "line": "..."} for each log line.
    """
    await ws.accept()
    current_task: Optional[asyncio.Task] = None
    current_container: Optional[str] = None

    async def stream_logs(cname: str):
        """Background task: read Docker logs and push to WebSocket."""
        try:
            async for line in _docker_log_generator(cname):
                for sub_line in line.strip().split("\n"):
                    if sub_line:
                        await ws.send_json({"type": "log", "line": sub_line})
        except (WebSocketDisconnect, Exception):
            pass

    try:
        # Default: stream active model's container if one is running
        if active_model and active_model in MODELS_CFG:
            cname = container_name(MODELS_CFG[active_model]["service"])
            current_container = cname
            current_task = asyncio.create_task(stream_logs(cname))
            await ws.send_json({"type": "info", "message": f"Streaming logs from {cname}"})

        while True:
            msg = await ws.receive_json()
            target = msg.get("container", "")
            if target and target != current_container:
                if current_task:
                    current_task.cancel()
                    try:
                        await current_task
                    except (asyncio.CancelledError, Exception):
                        pass
                current_container = target
                current_task = asyncio.create_task(stream_logs(target))
                await ws.send_json({"type": "info", "message": f"Streaming logs from {target}"})
    except WebSocketDisconnect:
        pass
    finally:
        if current_task:
            current_task.cancel()


# ---------------------------------------------------------------------------
# Proxy — catch-all routes (must be last)
# ---------------------------------------------------------------------------

@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_v1(request: Request, path: str):
    """
    Intercept all OpenAI-compatible API calls.

    1. Read body once.
    2. Resolve model field → config key or set key.
    3. Ensure the right model/set is active (swap if needed).
    4. Proxy to the correct sglang backend URL.
    """
    body = await request.body()

    model_field = ""
    if body:
        try:
            model_field = json.loads(body).get("model", "")
        except (json.JSONDecodeError, AttributeError):
            pass

    ready, base_url = await resolve_and_ensure(model_field)

    if not ready:
        if base_url is None and active_model is None and active_set is None:
            return JSONResponse(
                status_code=503,
                content={"error": {
                    "message": "No active model. Use /router/swap/{model} to load one first.",
                    "type": "no_active_model",
                }},
            )
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": f"Model '{model_field}' could not be loaded. Check router logs.",
                "type": "model_swap_error",
            }},
        )

    return await proxy_request(
        method=request.method,
        path=f"/v1/{path}",
        headers=dict(request.headers),
        body=body,
        params=dict(request.query_params),
        base_url=base_url,
    )


# ---------------------------------------------------------------------------
# Passthrough: non-/v1/ paths (e.g. /sleep, /wake_up, /health from sglang)
# ---------------------------------------------------------------------------

_BARE_OPENAI_INFERENCE_PATHS = frozenset({
    "completions",
    "chat/completions",
    "embeddings",
    "moderations",
    "audio/transcriptions",
    "audio/translations",
})


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_passthrough(request: Request, path: str):
    """
    Forward non-/v1/ paths directly to sglang.
    Rewrites bare OpenAI inference paths (missing /v1 prefix) with model routing.
    """
    body = await request.body()

    if path in _BARE_OPENAI_INFERENCE_PATHS:
        rewritten = f"/v1/{path}"
        log.debug("Bare OpenAI path /%s → rewriting to %s", path, rewritten)

        model_field = ""
        if body:
            try:
                model_field = json.loads(body).get("model", "")
            except (json.JSONDecodeError, AttributeError):
                pass

        ready, base_url = await resolve_and_ensure(model_field)

        if not ready:
            if base_url is None and active_model is None and active_set is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": {
                        "message": "No active model. Use /router/swap/{model} to load one first.",
                        "type": "no_active_model",
                    }},
                )
            return JSONResponse(
                status_code=503,
                content={"error": {
                    "message": f"Model '{model_field}' could not be loaded. Check router logs.",
                    "type": "model_swap_error",
                }},
            )

        return await proxy_request(
            method=request.method,
            path=rewritten,
            headers=dict(request.headers),
            body=body,
            params=dict(request.query_params),
            base_url=base_url,
        )

    # All other paths (sglang admin: /health, /sleep, /wake_up, /metrics, etc.)
    return await proxy_request(
        method=request.method,
        path=f"/{path}",
        headers=dict(request.headers),
        body=body,
        params=dict(request.query_params),
    )
