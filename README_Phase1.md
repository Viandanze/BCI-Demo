# NeuroDecode × LLM — Collaborative Reasoning (Phase 1)

> **Historical document** — Phase 1 era README. The project is now completed; see [README.md](README.md) for the final state.

BCI-LLM collaborative reasoning system: decode motor imagery into cognitive intents, generate LLM candidates, and let the user select via a second BCI round.

## Architecture

```
BrainFlow Synthetic Board (8ch, 250Hz)
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Preprocessing│────▶│  EEGNet /    │────▶│ Intent Encoder  │
│ (raw EEG)    │     │  Mock Decoder│     │ (MI → CogMode)  │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │ Context Manager │
                                          │ (state machine) │
                                          └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │   LLM Bridge    │
                                          │ (Ollama/API)    │
                                          └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │ Visual Feedback │
                                          │ (Flask + SSE)   │
                                          └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │ Second BCI Round│
                                          │ (select candidate)│
                                          └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │ Response Expand │
                                          │ + Context Update│
                                          └────────┬────────┘
                                                   │
                                              └─── loop ───┘
```

## Cognitive Mode Mapping

| Motor Imagery | Cognitive Mode | LLM Instruction |
|--------------|---------------|-----------------|
| Left Hand    |  QUERY      | Search for knowledge / factual information |
| Right Hand   |  REASON     | Logical reasoning / calculation / analysis |
| Feet         |  CREATE     | Generate creative solutions / novel ideas |
| Tongue       |  REVIEW     | Summarize / synthesize current context |

## Quick Start

### 1. Install dependencies

```bash
pip install brainflow flask numpy scipy
```

### 2. Run with Mock LLM (no Ollama needed)

```bash
python scripts/collaborative_reasoning_demo.py --backend mock
```

Open browser to `http://127.0.0.1:8080`

### 3. Run with Ollama (full experience)

```bash
# Install Ollama: https://ollama.ai
ollama pull qwen2.5:7b
ollama serve

# In another terminal:
python scripts/collaborative_reasoning_demo.py --backend ollama
```

### 4. Run with API (DeepSeek/OpenAI/etc)

```bash
python scripts/collaborative_reasoning_demo.py \
    --backend api \
    --api-url https://api.deepseek.com/v1 \
    --api-key YOUR_API_KEY \
    --api-model deepseek-chat
```

### 5. Run with Coze agent (bring your own bot)

The `coze` backend calls a deployed [Coze Coding](https://code.coze.cn) agent
that acts as a pure LLM relay. Bring your own bot in three steps:

1. **Create the relay agent** — on code.coze.cn, create an *agent* project
   with a prompt like: *"You are an LLM relay service. Forward any user text
   to the underlying model and return its reply verbatim, with no extra
   preamble or explanations."*
2. **Deploy it** — click *Deploy* on the project page and wait for the
   service to go live. Note the request domain (e.g. `https://xxxx.coze.site`)
   and the numeric project id shown on the deployment page.
3. **Configure credentials** — export environment variables (or copy
   `.env.example` to `.env`; never commit real tokens):

   ```bash
   export COZE_AGENT_DOMAIN="https://xxxx.coze.site"
   export COZE_PROJECT_ID="7600000000000000000"
   export COZE_API_TOKEN="pat_xxx"   # personal access token or project API token
   ```

Then run:

```bash
python scripts/collaborative_reasoning_demo.py --backend coze --real-decoder
```

Notes:
- Calls use the asynchronous task API (`async_run` + polling), which pairs
  well with `AsyncLLMBridge`.
- The agent sandbox may be reclaimed after ~1h of inactivity; the first
  request after idle can take noticeably longer (cold start). `poll_timeout`
  (default 120s) bounds the wait.
- Without the three credentials, the backend reports a visible
  `[Coze Error: missing configuration ...]` candidate instead of crashing.

### 6. Use trained EEGNet (instead of mock decoder)

Set `USE_REAL_DECODER = True` in `collaborative_reasoning_demo.py`,
or pass `--real-decoder` flag. Ensure model weights are at `models/eegnet_model.pth`.

## File Structure

```
NeuroDecode/
├── src/
│   ├── intent/
│   │   ├── __init__.py
│   │   ├── intent_encoder.py      # MI class → cognitive mode mapping
│   │   └── context_manager.py     # State machine + conversation context
│   ├── llm_bridge/
│   │   ├── __init__.py
│   │   ├── llm_client.py          # Ollama/API/Mock LLM backends
│   │   └── async_bridge.py        # Background worker: non-blocking LLM calls
│   └── feedback/
│       ├── __init__.py
│       └── visual_feedback.py     # Flask + SSE web display
├── scripts/
│   └── collaborative_reasoning_demo.py  # Main entry point
```

## State Machine

```
IDLE → DETECTING → INTENT_LOCKED → AWAITING_LLM
     → PRESENTING_CANDIDATES → SELECTING → COMPLETED → IDLE
```

- **IDLE**: Acquiring EEG, running decoder, waiting for intent to lock
- **INTENT_LOCKED**: Debounce passed, cognitive mode identified
- **AWAITING_LLM**: Querying LLM for candidate responses
- **PRESENTING_CANDIDATES**: Showing candidates, waiting for BCI selection
- **COMPLETED**: Selection made, response expanded, updating context

## Configuration

Key parameters in `collaborative_reasoning_demo.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WINDOW_SIZE` | 500 (2s) | EEG window length for decoding |
| `DEBOUNCE_FRAMES` | 3 | Consecutive same predictions to lock |
| `CONFIDENCE_THRESHOLD` | 0.5 | Min softmax probability to accept |
| `SELECTION_TIMEOUT` | 15s | Auto-select if no BCI response |

## Hardware Simulation

This Phase 1 uses **BrainFlow Synthetic Board** (zero hardware):
- Generates 8-channel synthetic EEG at 250Hz
- Includes realistic frequency components (alpha, beta, gamma)
- Single parameter switch to real hardware (OpenBCI, Cerelog, etc.)

To switch to real hardware, change board ID in `BrainFlowAcquisition`:
```python
# Synthetic (no hardware)
self.board_id = BoardIds.SYNTHETIC_BOARD

# OpenBCI Cyton (8 channels)
self.board_id = BoardIds.CYTON_BOARD

# g.Nautilus (32 channels)
self.board_id = BoardIds.NAUTILUS_BOARD
```

## Next Steps (Phase 2)

- [ ] Audio feedback (TTS via edge-tts)
- [ ] Context manager with multi-turn dialogue
- [ ] Real EEGNet decoder integration
- [ ] Streaming LLM responses (reduce perceived latency)
- [ ] Webots closed-loop demo (BCI → decode → LLM → robot action → feedback)
