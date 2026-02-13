# Computer Use Training — Roadmap

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
