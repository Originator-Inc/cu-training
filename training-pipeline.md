# The VLM Training Pipeline: Where Computer Use Fits

## TL;DR

There are **three major phases** of training (pre-, mid-, post-), but each has 2–4 internal stages, so the full ladder is ~7 steps. Computer use training is a **separate specialization layer applied at the very end**, after the model is already a fully post-trained VLM. It is not mixed into the general RLHF/DPO alignment loop.

---

## The Three Phases

| Phase | Purpose | Scale |
|-------|---------|-------|
| **Pre-training** | Learn language, world knowledge, basic reasoning | Trillions of tokens, months of compute |
| **Mid-training** | Refine capabilities, extend context, add domain depth | Hundreds of billions of tokens |
| **Post-training** | Align to instructions, preferences, and specialized tasks | Millions to billions of tokens, iterative |

---

## Phase 1: Pre-Training

**Objective:** Next-token prediction (autoregressive LM). For VLMs, also image-text contrastive/generative objectives.

**Data:** Trillions of tokens — web crawl, books, code, curated sources. Llama 3.1 used 15.6T tokens; Qwen 2 used 7T.

**What emerges:** Foundational language understanding, world knowledge, basic reasoning primitives. The model can complete text but cannot follow instructions or hold a conversation.

**Key techniques:** Data quality pipelines (filtering, dedup, classifier-based scoring), dynamic data mixing (adjusting web/code/math ratios during training), knowledge distillation for smaller models.

---

## Phase 2: Mid-Training

Formalized as a distinct phase around 2024–2025 ([Mid-Training Survey, arxiv 2510.06826](https://arxiv.org/abs/2510.06826)). Three sub-stages:

### 2a. Data Distribution Shift
Upweight code, math, chain-of-thought, synthetic textbooks. Downweight low-signal web crawl. Small fractions of domain-specific data produce disproportionate gains.

### 2b. Long-Context Extension
Dedicated stage extending from 4–8K to 32K–128K tokens using RoPE frequency remapping. Llama 3.1: 800B tokens across six progressive stages. Apple AFM: 100B tokens.

### 2c. Annealing
Final pre-training stage on very high-quality data with learning rate decay to zero, then checkpoint averaging. Llama 3.1: final 40M tokens with upsampled GSM8K and MATH data.

**What emerges:** Stronger reasoning, longer context, better code/math. Still a base model (not instruction-following), but a much better one.

**Key finding** ([Zhang et al., arxiv 2512.07783](https://arxiv.org/abs/2512.07783)): Mid-training installs priors that RL later exploits. RL cannot create capabilities the model has zero exposure to from pre/mid-training.

---

## Phase 3: Post-Training

This is where the base model becomes useful. It breaks down into **general alignment** and then **specialization**.

### 3a. General Alignment (iterative, 2–6 rounds)

The pattern every frontier lab follows:

```
Reward Model Training (human preference data)
    ↓
SFT Round 1 (human + synthetic instruction data)
    ↓
Preference Optimization Round 1 (DPO or RLHF-PPO or rejection sampling)
    ↓
... Repeat with fresh data from latest model (Llama 3 does 6 rounds) ...
    ↓
Reasoning RL (RLVR/GRPO on math/code with verifiable rewards)
```

**SFT** teaches instruction following on curated prompt/response pairs. **Preference optimization** steers the model toward human-preferred behaviors. These alternate iteratively — each round generates new data from the latest model.

### Types of RL used in general post-training

| Method | Reward Source | Used By |
|--------|-------------|---------|
| **RLHF (PPO)** | Learned reward model from human prefs | Anthropic, Apple, early OpenAI |
| **DPO** | Implicit from preference pairs | Meta (Llama 3), Qwen 2 |
| **RLVR (GRPO)** | Verifiable correctness (code compiles, math checks out) | DeepSeek-R1 |

**2025 trend:** GRPO has become dominant for reasoning (math, code) because it eliminates the reward model and critic entirely. For open-ended tasks (creative writing, helpfulness), learned reward models still needed.

### 3b. Computer Use Specialization (separate, at the end)

Computer use training requires a model that already has vision, instruction following, and basic reasoning. It is applied **after** general post-training is complete. Every frontier lab does this:

- **Anthropic:** Trained Claude on computer use using simple software (calculator, text editor) after the model was already fully post-trained. The skill generalized to complex software.
- **OpenAI:** Two stages on top of GPT-4o — supervised learning for perception/control, then RL (similar to o1/o3 techniques) for reasoning and error correction.
- **ByteDance (UI-TARS-2):** Starts from Seed-thinking-1.6 (already a reasoning model), applies continual pre-training → SFT → RL specifically for GUI tasks.
- **ComputerRL:** Starts from GLM-4-9B (already instruction-tuned), applies behavior cloning then online RL.

**The pattern is clear:** take a fully trained, already-aligned VLM and add a specialized training stack on top.

---

## The Computer Use Training Sub-Pipeline

Once you have a post-trained VLM, computer use training has its own internal stages:

```
General Post-Trained VLM
    ↓
[A] Perception Training — GUI element recognition, dense captioning, state understanding
    ↓
[B] Action Modeling / Behavior Cloning (SFT) — supervised on human demonstration trajectories
    ↓
[C] Reasoning Integration — chain-of-thought for GUI (task decomposition, reflection)
    ↓
[D] Online Multi-Turn RL — GRPO or PPO in parallel VMs, binary task-completion rewards
    ↓
[E] Iterative Refinement — Entropulse cycles, DPO on error-correction trajectories
```

**Stage D is the hard part.** Multi-turn RL (30+ actions, single binary reward at end) is much harder than single-turn RLVR because of:
- **Credit assignment** — which of 30 clicks was the mistake?
- **Entropy collapse** — policy collapses to repetitive actions over long rollouts
- **Environment parallelism** — need thousands of VMs running simultaneously

Solutions from frontier work:
- **ComputerRL:** GRPO + Entropulse (alternate RL/SFT when training stagnates). SOTA 48.9% on OSWorld.
- **UI-TARS-2:** PPO with decoupled GAE, length-adaptive GAE, value pretraining. 47.5% on OSWorld. Finding: PPO > GRPO for multi-turn.
- **DART-GUI:** Decoupled async RL, trains selectively on high-entropy steps. 42.1% on OSWorld with just 7B model.

---

## The Full Training Ladder (Summary Diagram)

```
═══════════════════════════════════════════════════════════════
PHASE 1: PRE-TRAINING                    [Trillions of tokens]
═══════════════════════════════════════════════════════════════
  1. Core pre-training (next-token prediction on web/books/code)
  2. Continued pre-training (upweight code/math/synthetic)
  3. Context lengthening (RoPE scaling, progressive stages)
  4. Annealing (high-quality data, LR decay, checkpoint averaging)

═══════════════════════════════════════════════════════════════
PHASE 2: POST-TRAINING (GENERAL)         [Iterative, 2–6 rounds]
═══════════════════════════════════════════════════════════════
  5. SFT (instruction following)
  6. Preference optimization (DPO / RLHF-PPO / rejection sampling)
     ... repeat rounds 5–6 with fresh data ...
  7. Reasoning RL (RLVR/GRPO on math/code)

═══════════════════════════════════════════════════════════════
PHASE 3: COMPUTER USE SPECIALIZATION     [Thousands of VM rollouts]
═══════════════════════════════════════════════════════════════
  8. Perception training (GUI screenshots, grounding)
  9. Behavior cloning / SFT (demonstration trajectories)
 10. Reasoning integration (CoT for GUI tasks)
 11. Online multi-turn RL (GRPO or PPO in parallel VMs)
 12. Iterative refinement (Entropulse, DPO on error data)
```

---

## Key Takeaways

1. **Three phases, ~12 steps.** Pre-training creates primitives. Mid-training amplifies them. Post-training aligns and specializes. Each phase contributes something the others cannot.

2. **Post-training is SFT then RL, iteratively.** Every frontier lab follows: SFT → preference optimization → repeat. The RL flavors differ (DPO vs PPO vs GRPO) but the structure is the same.

3. **Computer use is a separate specialization at the very end.** It is not mixed into general alignment. It requires a fully post-trained VLM as starting point, then applies its own perception → SFT → RL pipeline.

4. **The RL for computer use is multi-turn GRPO/PPO in simulated environments.** The "verifiable reward" is task completion in a VM. The main challenges are credit assignment, entropy collapse, and parallelizing thousands of environments.

---

## Sources

- [Raschka, "New LLM Pre-training and Post-training Paradigms" (Aug 2024)](https://sebastianraschka.com/blog/2024/new-llm-pre-training-and-post-training.html)
- [Raschka, "The State of RL for LLM Reasoning" (2025)](https://sebastianraschka.com/blog/2025/the-state-of-reinforcement-learning-for-llm-reasoning.html)
- [The Llama 3 Herd of Models (Meta, Jul 2024)](https://arxiv.org/abs/2407.21783)
- [UI-TARS (ByteDance, Jan 2025)](https://arxiv.org/abs/2501.12326)
- [UI-TARS-2 (ByteDance, Sep 2025)](https://arxiv.org/abs/2509.02544)
- [ComputerRL (Zhipu AI, Aug 2025)](https://arxiv.org/abs/2508.14040)
- [Mid-Training Survey (Oct 2025)](https://arxiv.org/abs/2510.06826)
- [On the Interplay of Pre-Training, Mid-Training, and RL (Dec 2025)](https://arxiv.org/abs/2512.07783)
- [Anthropic, "Developing a computer use model"](https://www.anthropic.com/news/developing-computer-use)
- [Anthropic, "Constitutional AI: Harmlessness from AI Feedback" (Dec 2022)](https://arxiv.org/abs/2212.08073)
- [Anthropic, Claude 3 Model Card](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)
- [OpenAI, "Computer-Using Agent"](https://openai.com/index/computer-using-agent/)
