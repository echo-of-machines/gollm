# GoLLM

**Multi-backend local LLM container manager with browser GUI.**

GoLLM manages local LLM inference containers through a single OpenAI-compatible API endpoint. Install models from HuggingFace, swap between them with one click, and monitor everything from a browser dashboard.

## Features

- **Multi-backend** — SGLang and vLLM support, with backend templates for easy model installation
- **One-click model swap** — stop current model, start new one, route traffic automatically
- **Browser GUI** — manage services, models, sets, and jobs from `http://localhost:30000/ui/`
- **Live terminal** — WebSocket-powered Docker log streaming in the browser
- **RAM-aware** — checks available memory before loading models to prevent OOM
- **OpenAI-compatible API** — drop-in replacement at `/v1/chat/completions`
- **HuggingFace integration** — download models directly from the GUI with token management
- **Model sets** — group models to run simultaneously with combined RAM checks

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/gollm.git
cd gollm

# 2. Configure (optional)
cp .env.example .env
# Edit .env to set your HF_TOKEN and port

# 3. Start GoLLM
docker compose up -d --build model-router

# 4. Open the GUI
open http://localhost:30000/ui/
```

## Requirements

- Docker with Compose plugin
- NVIDIA GPU with drivers installed
- NVIDIA Container Toolkit (`nvidia-docker`)

## Installing Models

### Via GUI
1. Open `http://localhost:30000/ui/`
2. Go to **Models** tab → click **+ Install**
3. Enter a name, select backend (SGLang/vLLM), paste the HuggingFace model path
4. Click **Install** — weights download in the background

### Via API
```bash
curl -X POST http://localhost:30000/router/models/install \
  -H "Content-Type: application/json" \
  -d '{
    "key": "qwen3.5",
    "backend": "sglang",
    "model_path": "Qwen/Qwen3.5-35B-A3B-FP8",
    "download": true
  }'
```

## Loading & Swapping Models

### Via GUI
Models tab → click **Load** on the model card.

### Via API
```bash
# Load a specific model
curl -X POST http://localhost:30000/router/swap/qwen3.5

# Send inference (auto-swaps if needed)
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `GOLLM_PORT` | `30000` | Host port for the GoLLM API and GUI |
| `HF_TOKEN` | (empty) | HuggingFace token for gated model downloads |

### Config File (`model-router/config.yaml`)

Models, sets, and router settings are stored here. Edited automatically by the GUI, or manually with hot-reload via the Status tab.

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Router health check |
| `/v1/chat/completions` | POST | OpenAI-compatible inference |
| `/v1/models` | GET | List available models |
| `/router/swap/{key}` | POST | Load/swap to a model |
| `/router/models` | GET | List registered models with status |
| `/router/models/install` | POST | Install a new model |
| `/router/models/{key}/stop` | POST | Stop a model container |
| `/router/backends` | GET | List available backends |
| `/router/system` | GET | System status (RAM, active model) |
| `/router/services` | GET | List Docker Compose services |
| `/router/sets` | GET/POST | Manage model sets |
| `/router/jobs` | GET | List download jobs |
| `/router/hf-token` | GET/POST/DELETE | Manage HuggingFace token |
| `/router/config/reload` | POST | Hot-reload config.yaml |
| `/ws/logs` | WebSocket | Live container log stream |
| `/ui/` | GET | Browser GUI |

## Systemd Service (Optional)

To start GoLLM on boot:

```bash
# Edit gollm.service — update WorkingDirectory to your install path
sudo cp gollm.service /etc/systemd/system/
sudo systemctl enable gollm
sudo systemctl start gollm
```

## Architecture

```
Clients (curl, apps) → GoLLM:30000 → Active backend (SGLang/vLLM)
                            ↕
                    Docker socket (container lifecycle)
```

GoLLM runs as a Docker container that manages other containers via the Docker socket. When you load a model, it starts the appropriate backend container and proxies traffic to it.

## License

MIT
