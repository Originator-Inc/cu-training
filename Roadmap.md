# Computer Use Training — Roadmap

## 5-Day Research Program (~2 hrs/day)

### Day 1 — Foundations: Where Computer Use Fits in the Training Stack
- **Read:** Raschka's "New LLM Pre-training and Post-training Paradigms" — map the full pre-train → mid-train → SFT → RLHF → RLVR pipeline
- **Read:** DeepSeek-R1 (sections 2–3 only) — understand GRPO and single-turn RLVR
- **Goal:** Be able to draw the training ladder from scratch and explain why multi-turn agent RL is harder than single-turn

### Day 2 — Core Papers: How Computer Use RL Actually Works
- **Read:** ComputerRL paper — full pipeline, Entropulse trick, API-GUI paradigm
- **Read:** WebAgent-R1 — minimal viable multi-turn GRPO (binary rewards, no reward model)
- **Skim:** WebRL if time allows — curriculum learning and outcome reward models
- **Goal:** Understand the two main approaches (minimal GRPO vs. full infrastructure) and their tradeoffs

### Day 3 — Hands-On: Run Claude CUA and Qwen3-VL Locally
- **Set up:** Docker VM with VNC (`anthropic-quickstarts:computer-use-demo`)
- **Test:** Claude CUA on 2–3 manual tasks via the web UI — observe how it plans and acts
- **Test:** Serve Qwen3-VL-8B via vLLM, run against the same tasks (or use OpenCUA wrapper)
- **Goal:** First-hand feel for where each model succeeds/fails, action latency, token costs

### Day 4 — Evaluation: OSWorld and Side-by-Side Comparison
- **Set up:** OSWorld with a small task subset (5–10 tasks)
- **Run:** Claude and Qwen3-VL against the same tasks, collect trajectories
- **Review:** Compare trajectories side-by-side (screenshots + actions in result dirs)
- **Goal:** Quantify the gap — where does Qwen3-VL fail that Claude succeeds? What patterns emerge?

### Day 5 — SDPO & Training Plan: Chart the Path Forward
- **Read:** UI-TARS-2 and DART-GUI papers — PPO stabilization tricks, decoupled async training
- **Review:** verl → verl-agent → SDPO repo stack, read the example scripts
- **Draft:** Concrete fine-tuning plan — what data, what hardware, what the self-teacher prompt looks like for GUI failures
- **Goal:** Have a written 1-pager on "how we'd run SDPO on Qwen3-VL for computer use" ready for team review

---

## Pre-Reading / Research

Here's the absolute essentials, in reading order:

### Understanding the full training pipeline (so you know where computer use fits)

1. **Sebastian Raschka — "New LLM Pre-training and Post-training Paradigms" (Aug 2024)** — Gives you the pre-train → mid-train → SFT → RLHF → RLVR stack in one post. You need this foundation. The computer use stage sits at the end as a specialized form of multi-turn RL post-training.

2. **DeepSeek-R1 paper (Jan 2025)** — [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948) — Read sections 2 and 3 only. You need to understand GRPO and how single-turn RLVR works before you can understand why multi-turn agent RL is harder. This is the baseline everything else extends from.

### How computer use training actually works

3. **ComputerRL paper (Aug 2025)** — [arxiv.org/abs/2508.14040](https://arxiv.org/abs/2508.14040) — This is the single most important paper. Current SOTA on OSWorld (48.1%). Covers the full pipeline: SFT warm-up from trajectory data → online RL in parallel virtual desktops → the Entropulse trick (alternating RL/SFT to prevent entropy collapse). Also introduces the API-GUI paradigm which is how frontier agents actually operate — they don't just click pixels, they mix programmatic calls with GUI actions.

4. **WebAgent-R1 (May 2025)** — [arxiv.org/abs/2505.16421](https://arxiv.org/abs/2505.16421) — The cleanest demonstration of pure end-to-end multi-turn RL for browser use. No auxiliary reward models, no replay buffers — just binary task-success rewards and on-policy GRPO. Read this to understand the minimal viable approach.

5. **WebRL (ICLR 2025)** — [arxiv.org/abs/2411.02337](https://arxiv.org/abs/2411.02337) — Read if you have time. Adds curriculum learning (auto-generating harder tasks from failures) and a trained outcome reward model. More complex than WebAgent-R1 but shows what you gain from richer training infrastructure.

That's it. Five items. The reading order matters: 1 gives you the map, 2 gives you the RL fundamentals, 3–4 are the actual computer use training methods, 5 is bonus depth.

> **Core mental model:** computer use training = multi-turn GRPO where the "verifiable reward" is task completion in a simulated environment (VM or browser), and the main engineering challenges are parallelizing thousands of environments, preventing entropy collapse over long rollouts, and designing action spaces that mix API calls with GUI interactions.

---

## How are computer use models trained?

- What architectures and training paradigms are used (e.g. action-prediction heads on top of VLMs, set-of-marks, direct pixel-to-action)?
- What data is used — synthetic trajectories, human demonstrations, or a mix?
- Which trajectory selection strategies work best (reward-filtered, DPO pairs, success-only)?
- How would we tune these models internally on our own environments and tasks?

## Open source vs. private models

- How good or bad are open-source computer use models compared to private/proprietary ones?
- What benchmarks exist (OSWorld, WebArena, ScreenSpot, etc.) and where do open models fall short?
- Can we test open-source models in a local tool or harness for rapid evaluation?

## Where does computer use training fit in the overall VLM pipeline?

- Where in post-training does computer use capability get added (SFT, RLHF, tool-use alignment)?
- What kind of pre-training or mid-training is required for a VLM to do well at computer use (e.g. UI understanding, OCR, spatial grounding)?
- How does the training ladder look: pre-train → mid-train → SFT → RL → computer-use specialization?

---

## Setting Up Qwen3-VL & Claude Computer Use for Evaluation and Fine-Tuning

### Infrastructure: Shared Environment

Both agents need a desktop environment to interact with. Use **OSWorld** as the common evaluation harness — it spins up Ubuntu VMs with real applications (Chrome, LibreOffice, VS Code, etc.) and provides task definitions with automated success checks.

```bash
git clone https://github.com/xlang-ai/OSWorld && cd OSWorld
pip install -r requirements.txt
# Provision VMs (supports AWS, Azure, local Docker)
python run_multienv.py --provider_name aws --num_envs 5
```

For lighter local testing, use **Docker-based VMs** with VNC access so you can visually observe each agent in real time:

```bash
docker run -d -p 5900:5900 -p 6080:6080 ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo
# Open http://localhost:6080 in your browser to watch the agent
```

---

### Agent 1: Claude Computer Use

Claude's CUA is API-driven — you send screenshots and receive action commands. Use Anthropic's reference implementation:

```bash
git clone https://github.com/anthropics/anthropic-quickstarts
cd anthropic-quickstarts/computer-use-demo
export ANTHROPIC_API_KEY=sk-ant-...
docker compose up
```

This gives you a web UI at `localhost:8080` where you can issue tasks and watch Claude interact with the VM in real time. For programmatic evaluation against OSWorld tasks, use the OSWorld Claude integration:

```bash
python run_multienv.py \
  --observation_type screenshot \
  --model claude-sonnet-4-5-20250929 \
  --max_steps 50 \
  --result_dir ./results/claude
```

---

### Agent 2: Qwen3-VL (via OpenCUA or Direct)

**Option A — Run OpenCUA models** (pre-trained CUA on top of Qwen2.5-VL):

```bash
# Serve with vLLM (OpenCUA-7B fits on 1x A100, 32B needs 4x)
vllm serve xlangai/OpenCUA-7B \
  --trust-remote-code \
  --served-model-name opencua-7b \
  --host 0.0.0.0 --port 8000

# Evaluate on OSWorld
python run_multienv_opencua.py \
  --observation_type screenshot \
  --model OpenCUA-7B \
  --max_steps 100 \
  --max_history_turns 4 \
  --num_envs 5 \
  --result_dir ./results/opencua
```

**Option B — Run Qwen3-VL directly** (base VLM, no CUA fine-tuning):

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000

# You'll need a wrapper to format screenshots + actions into Qwen3-VL's chat format
# See the OSWorld Qwen integration or write a simple agent loop
```

**Key parameter notes:**
- `--max_history_turns 4` — limits visual history to 4 past screenshots (token budget management; each 1920×1080 screenshot costs thousands of tokens)
- The original task instruction always remains in the prompt; only intermediate step history is windowed
- Qwen3-VL supports 256K context natively, extendable to 1M with YaRN

---

### Visual / Manual Evaluation

For side-by-side manual comparison:

1. **VNC viewer** — connect to the VM's VNC port (5900) to watch actions live
2. **Trajectory recording** — both OSWorld and OpenCUA save per-step screenshots + actions to the result directory. Review them post-hoc.
3. **AgentNetBench** (offline) — if you want fast iteration without spinning up VMs, OpenCUA's offline benchmark compares predicted actions against gold-standard trajectories:

```bash
cd OpenCUA
python eval/agentnet_bench.py --model_output ./results/opencua --gold_data ./data/agentnet_bench
```

---

### Fine-Tuning Qwen3-VL for Computer Use

**Data sources:**
- **AgentNet** (OpenCUA's dataset) — 3 OSes, 200+ apps/websites, with reflective CoT annotations. Available on HuggingFace under `xlangai/`.
- **Your own trajectories** — use OpenCUA's AgentNetTool to record demonstrations on your own machine, or export trajectories from OSWorld evaluation runs.

**Training with Unsloth** (most accessible, free Colab option for 8B):

```bash
pip install unsloth --break-system-packages

# Use the Unsloth Qwen3-VL fine-tuning notebook:
# https://docs.unsloth.ai/models/qwen3-vl-run-and-fine-tune
# Supports SFT on image/video data, 1.7x faster, 60% less VRAM
```

**Training with OpenCUA's recipe** (for reproducing their results):

OpenCUA uses a 2-stage SFT approach on Qwen2.5-VL:
- **Stage 1**: Grounding + understanding (grounding trajectories, tutorials, captions, general VL tasks) — 40B tokens
- **Stage 2**: CUA planning (45% planning, 20% grounding, rest general) — shorter

The training code is based on internal Kimi infrastructure; the open-source version provides data processing scripts and the data itself. For your own fine-tuning, the practical path is:

```bash
# 1. Download AgentNet data
huggingface-cli download xlangai/AgentNet --local-dir ./agentnet_data

# 2. Process into SFT format (state-action pairs with CoT)
python data/process_trajectories.py --input ./agentnet_data --output ./sft_data

# 3. Fine-tune with your preferred framework (Unsloth, axolotl, or transformers)
# Key settings: multi-image input, L2 CoT format (Thought + Action),
# mixed data (CUA trajectories + general VL to prevent catastrophic forgetting)
```

**RL fine-tuning** (advanced): For RL on top of SFT, look at **EvoCUA** (Meituan) which achieved 56.7% on OSWorld-Verified using RL-augmented training on top of open-source VLMs, or **WebAgent-R1** for browser-specific multi-turn GRPO.

---

## SDPO for Computer Use — Quick Reference

### 1. Base Model

**Qwen3-VL-8B** is the recommended starting point — it's the latest in the Qwen VL series, already supported by verl-agent, and inherits the computer use and phone use capabilities from Qwen2.5-VL (grounding, agentic tool use, screenshot understanding).

- Qwen3-VL: [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- Fallback / reference: [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [32B](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct), [72B](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)

### 2. Training Framework Stack

You need three layers:

**verl** (core RL training) → [github.com/verl-project/verl](https://github.com/verl-project/verl)
- Has a ready-made VLM GRPO example script: `run_qwen2_5_vl-7b.sh`
- Handles FSDP2 training + vLLM/SGLang rollouts, LoRA support, VLM RL natively

**verl-agent** (multi-turn agentic RL extension) → [github.com/langfengQ/verl-agent](https://github.com/langfengQ/verl-agent)
- Built on verl, adds multi-turn interaction loops, customizable memory modules, per-step input structure
- Already supports Qwen2.5-VL and Qwen3-VL with LoRA (7B on 2× H100)
- Includes GUI environments: WebShop, AppWorld, Sokoban, Gym Cards
- Implements GRPO, PPO, DAPO, GiGPO and more

**SDPO repo** (self-distillation logic) → [github.com/lasgroup/SDPO](https://github.com/lasgroup/SDPO)
- Built on verl, swaps advantage computation from GRPO to self-distillation
- You'd need to port/merge the SDPO advantage computation into verl-agent's multi-turn loop

### 3. Environment / Benchmark

**OSWorld** → [github.com/xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld), [os-world.github.io](https://os-world.github.io)
- 369 real computer tasks on Ubuntu VMs (Docker-based), execution-based evaluation
- Supports parallel VM rollouts for RL training
- The standard benchmark — UI-TARS-2 trains against it, DART-GUI evaluates on it

### 4. Key Papers to Read

These represent the current frontier of RL for GUI agents:

- **UI-TARS-2** ([arxiv.org/abs/2509.02544](https://arxiv.org/abs/2509.02544)) — The most comprehensive work. Multi-turn PPO with stabilization tricks: decoupled GAE, length-adaptive GAE, value pretraining, reward shaping. Async rollouts on VMs. SOTA on OSWorld (47.5%). Findings: PPO > GRPO for multi-turn, VLM-as-verifier works for agent tasks.
- **DART-GUI** ([arxiv.org/html/2509.23866v1](https://arxiv.org/html/2509.23866v1)) — Decoupled async RL training. 42.13% on OSWorld with just 7B model. Trains selectively on high-entropy steps, uses pre-collected successful trajectories to bootstrap.
- **ARPO** — Extends GRPO with replay buffer for GUI agents, task selection strategy for stability.
- **ZeroGUI** — No human labels needed. Uses VLMs to generate tasks AND evaluate success, two-stage RL. +14% on UI-TARS baseline.
- **GUI agent paper list** → [github.com/OSU-NLP-Group/GUI-Agents-Paper-List](https://github.com/OSU-NLP-Group/GUI-Agents-Paper-List) — Comprehensive and actively maintained.

### 5. Why SDPO Is Particularly Interesting Here

The SDPO paper is LLM-only, but the approach maps naturally to computer use because:

- **Rich environment feedback is already available** — screenshots show what happened after an action, error dialogs, changed UI state. This is exactly the kind of feedback SDPO's self-teacher conditions on.
- **Dense credit assignment matters more in multi-step GUI tasks** — GRPO gives the same reward to every action in a 30-step trajectory. SDPO can identify which specific click or type was the mistake by showing the model the failure screenshot and asking "where did it go wrong?"
- **Concise action generation** — SDPO's biggest qualitative win was generating shorter, more direct reasoning. GUI agents notoriously waste tokens on verbose "thinking" that doesn't help grounding.

### 6. Practical Sketch of the Integration

The rough plan would be:

1. Start with verl-agent's multi-turn GRPO setup with Qwen3-VL-8B on OSWorld
2. Get baseline working with binary task-completion rewards
3. Port SDPO's advantage computation from the `lasgroup/SDPO` repo into verl-agent's trainer
4. Design the self-teacher template: on failed trajectories, feed the model its own action history + final failure screenshot/state + any error messages, and compute logit divergences to identify where the policy should have acted differently
5. Start with LoRA (feasible on 2× H100 per verl-agent), scale to full fine-tuning if results warrant it

The main engineering challenge is **step 4** — designing the feedback-conditioned self-teacher prompt for a VLM. The SDPO paper only did this for text (code errors, test failures). For GUI tasks you'd condition on the failure screenshot and/or accessibility tree diff, which is a novel but natural extension.
