# GoLLM

**Multi-backend local LLM container manager with browser GUI.**

GoLLM manages local LLM inference containers through a single OpenAI-compatible API endpoint. Install models from HuggingFace, swap between them with one click, and monitor everything from a browser dashboard.

## Features

- **Multi-backend** — SGLang, vLLM, and Custom Docker image support
- **One-click model swap** — stop current model, start new one, route traffic automatically
- **Browser GUI** — manage services, models, sets, and jobs from `http://localhost:30000/ui/`
- **Live terminal** — WebSocket-powered Docker log streaming in the browser
- **RAM-aware** — checks available memory before loading models to prevent OOM
- **OpenAI-compatible API** — drop-in replacement at `/v1/chat/completions`
- **HuggingFace integration** — download models directly, with in-app token management
- **Model sets** — group models to run simultaneously with combined RAM checks

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/echo-of-machines/gollm.git
cd gollm

# 2. Configure (optional)
cp .env.example .env
# Edit .env to set your HF_TOKEN and port

# 3. Start GoLLM
docker compose up -d --build model-router

# 4. Open the GUI
xdg-open http://localhost:30000/ui/   # Linux
open http://localhost:30000/ui/        # macOS
```

## Requirements

- Docker with Compose plugin
- NVIDIA GPU with drivers installed
- NVIDIA Container Toolkit

## For AI Agents & Automation

GoLLM exposes a standard OpenAI-compatible API. Point any tool that supports OpenAI's API format to `http://localhost:30000` and it works as a drop-in replacement.

### Integration Example

```python
# Works with any OpenAI-compatible client
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="unused")

# List available models
models = client.models.list()

# Chat completion — GoLLM auto-loads the requested model
response = client.chat.completions.create(
    model="qwen3.5",  # any registered model name or alias
    messages=[{"role": "user", "content": "Hello"}],
)
```

### curl Examples

```bash
# Check what's running
curl http://localhost:30000/health
# → {"status": "ok", "active_model": "qwen3.5", "active_set": null}

# List registered models
curl http://localhost:30000/router/models
# → {"models": [{"key": "qwen3.5", "container_status": "running", "active": true, ...}]}

# Load a specific model (stops current, starts target)
curl -X POST http://localhost:30000/router/swap/nemotron-nano

# Send inference (auto-swaps to requested model if needed)
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5", "messages": [{"role": "user", "content": "Hello"}]}'

# Install a new model (downloads weights, registers in config)
curl -X POST http://localhost:30000/router/models/install \
  -H "Content-Type: application/json" \
  -d '{
    "key": "llama4",
    "backend": "vllm",
    "model_path": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "download": true
  }'

# Install with custom Docker image (e.g. model-specific vLLM build)
curl -X POST http://localhost:30000/router/models/install \
  -H "Content-Type: application/json" \
  -d '{
    "key": "gemma4",
    "backend": "custom",
    "model_path": "google/gemma-4-26B-A4B-it",
    "image": "vllm/vllm-openai:gemma4-cu130",
    "download": true
  }'

# Set HuggingFace token (required for gated models)
curl -X POST http://localhost:30000/router/hf-token \
  -H "Content-Type: application/json" \
  -d '{"token": "hf_your_token_here"}'

# Check system status (RAM, active model)
curl http://localhost:30000/router/system
# → {"active_model": "qwen3.5", "ram_used_gb": 45.2, "ram_total_gb": 122.0, ...}

# Stop a model
curl -X POST http://localhost:30000/router/models/qwen3.5/stop
```

### Model Swap Behavior

When a request arrives with a `model` field that differs from the currently active model:

1. In-flight requests to the current model are drained
2. Current model container is stopped (frees GPU memory)
3. Target model container is started
4. Health check polls until the model is ready
5. Request is proxied to the new model

This means you can send requests for any registered model — GoLLM handles the swap transparently. Cold starts take 3-5 minutes depending on model size.

### Aliases

Each model can have multiple aliases. For example, a model registered as `qwen3.5` might also respond to `qwen3.5-35b`, `Qwen/Qwen3.5-35B-A3B-FP8`, or `default`. Aliases are configured during install or in `config.yaml`.

## Installing Models

### Via GUI
1. Open `http://localhost:30000/ui/`
2. Go to **Models** tab, click **+ Install**
3. Enter a name, select backend (SGLang / vLLM / Custom), paste the HuggingFace model path
4. For Custom backend, also provide the Docker image (e.g. `vllm/vllm-openai:gemma4-cu130`)
5. Click **Install** — weights download in the background, model card appears when ready

### Via API
See curl examples above.

### Backends

| Backend | Image | Port | Use Case |
|---------|-------|------|----------|
| **SGLang** | `lmsysorg/sglang:dev-cu13` | 30000 | High-performance inference, MoE models |
| **vLLM** | `vllm/vllm-openai:latest` | 8000 | Wide model support, OpenAI-compatible |
| **Custom** | User-provided | 8000 | Model-specific images, experimental builds |

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `GOLLM_PORT` | `30000` | Host port for the GoLLM API and GUI |
| `HF_TOKEN` | (empty) | HuggingFace token for gated model downloads |

### Config File (`model-router/config.yaml`)

Models, sets, and router settings are stored here. Edited automatically by the GUI/API, or manually with hot-reload via the Status tab.

```yaml
models:
  qwen3.5:
    service: sglang-qwen
    base_url: "http://sglang-qwen:30000"
    model_path: Qwen/Qwen3.5-35B-A3B-FP8
    aliases: [qwen3.5, default]
    health_timeout_start: 300

router:
  compose_project: gollm
  health_poll_interval: 2
  in_flight_drain_timeout: 120
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Router health check |
| `/v1/chat/completions` | POST | OpenAI-compatible chat inference |
| `/v1/completions` | POST | OpenAI-compatible text completion |
| `/v1/models` | GET | List available models (OpenAI format) |
| `/router/swap/{key}` | POST | Load/swap to a model |
| `/router/models` | GET | List registered models with container status |
| `/router/models/install` | POST | Install a new model (compose + config + download) |
| `/router/models/{key}` | DELETE | Unregister a model |
| `/router/models/{key}/stop` | POST | Stop a model container |
| `/router/models/{key}/start` | POST | Start a model container |
| `/router/backends` | GET | List available backend templates |
| `/router/system` | GET | System status (RAM, active model, swap state) |
| `/router/services` | GET | List all Docker Compose services |
| `/router/services/{svc}/start` | POST | Start a compose service |
| `/router/services/{svc}/stop` | POST | Stop a compose service |
| `/router/services/{svc}/restart` | POST | Restart a compose service |
| `/router/sets` | GET/POST | List or create model sets |
| `/router/sets/{key}` | DELETE | Delete a model set |
| `/router/sets/{key}/start` | POST | Start all models in a set |
| `/router/sets/{key}/stop` | POST | Stop all models in a set |
| `/router/jobs` | GET | List active/completed jobs |
| `/router/jobs/{id}/cancel` | POST | Cancel a running job |
| `/router/hf-token` | GET/POST/DELETE | Manage HuggingFace token |
| `/router/config/reload` | POST | Hot-reload config.yaml |
| `/ws/logs` | WebSocket | Live container log stream |
| `/ui/` | GET | Browser GUI |

## Systemd Service (Optional)

To auto-start GoLLM on boot:

```bash
# Edit gollm.service — update WorkingDirectory to your install path
sudo cp gollm.service /etc/systemd/system/
sudo systemctl enable gollm
sudo systemctl start gollm
```

## Architecture

```
Clients (curl, apps, AI agents)
         │
         ▼
   GoLLM Router (:30000)
   ┌─────────────────────┐
   │  OpenAI-compat API  │  ← /v1/chat/completions, /v1/models
   │  Management API     │  ← /router/*, install, swap, status
   │  Browser GUI        │  ← /ui/
   │  WebSocket logs     │  ← /ws/logs
   └────────┬────────────┘
            │ Docker socket
            ▼
   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
   │  SGLang :30000  │  │  vLLM :8000    │  │  Custom :port  │
   │  (Qwen, etc.)  │  │  (Llama, etc.) │  │  (any image)   │
   └────────────────┘  └────────────────┘  └────────────────┘
         ▲                    ▲                    ▲
         └────────────────────┴────────────────────┘
                    Shared HuggingFace cache volume
```

GoLLM runs as a Docker container that manages other containers via the Docker socket. Only one model (or one set of models) is active at a time. When you load a model, GoLLM stops the current one, starts the new one, and proxies all traffic transparently.

## License

MIT
