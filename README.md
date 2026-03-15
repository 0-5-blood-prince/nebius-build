# RoboMind

LLM-orchestrated robot control with predictive planning. Talk to a Unitree G1 humanoid in plain English — the LLM reasons about what to do, predicts the outcome using parallel MuJoCo simulations, and executes the action if it's safe.

Built at the Nebius AI hackathon.

---

## How it works

```
User instruction (text)
        │
        ▼
┌───────────────┐     robot state (JSON)     ┌─────────────────┐
│  MuJoCo / G1  │ ────────────────────────▶  │   Nebius LLM    │
│  Simulation   │                             │  (Qwen3-235B)   │
└───────────────┘ ◀────── action command ──── └─────────────────┘
        │                                              │
        │                                    observation + reasoning
        │                                    + predicted outcome
        ▼
┌───────────────┐
│  World Model  │  runs N parallel rollouts from current state
│  (MuJoCo)    │  before executing — aborts if risk is HIGH
└───────────────┘
```

**Three components:**

1. **Perception** — `robot_env.py` extracts position, velocity, orientation, and stability from MuJoCo state and converts it to structured JSON the LLM can read.

2. **Reasoning** — `llm_brain.py` sends the state + user instruction to the Nebius Token Factory API and gets back an observation, multi-step reasoning chain, predicted outcome, action choice, and confidence score — all as structured JSON.

3. **World model** — `world_model.py` runs 5 noisy parallel MuJoCo rollouts for the chosen action before executing it. If stability rate drops below 50% the system flags HIGH risk and asks the LLM to re-plan.

---

## Setup

```bash
git clone <this repo>
cd robomind

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Clone the G1 robot model
git clone --depth=1 https://github.com/google-deepmind/mujoco_menagerie

# Verify G1 loads
python robot_env.py
```

Get a Nebius Token Factory API key at https://tokenfactory.nebius.com/ and export it:

```bash
export NEBIUS_API_KEY=your_key_here
```

---

## Running

```bash
# Full demo with live LLM reasoning
.venv/bin/mjpython demo.py

# Stub mode — no API key needed, uses pre-scripted decisions
.venv/bin/mjpython demo.py --stub
```

> **macOS note:** MuJoCo's passive viewer requires `mjpython` on macOS. It ships with the `mujoco` pip package at `.venv/bin/mjpython`.

The viewer opens showing the full G1 robot. The terminal shows the LLM's live reasoning chain for each step.

---

## Demo sequence

| Step | Instruction | Action |
|------|-------------|--------|
| 1 | Walk forward to explore | `walk_forward` |
| 2 | Stop and assess stability | `stop` |
| 3 | Do the Shah Rukh Khan pose | `srk_pose` |
| 4 | Wave to the audience | `wave` |

---

## Available actions

| Action | Description |
|--------|-------------|
| `walk_forward` | Drive hip pitch + knee joints forward |
| `walk_backward` | Same, reversed |
| `turn_left` | Differential hip yaw |
| `turn_right` | Differential hip yaw, opposite |
| `stop` / `stand_up` | Neutral joint targets |
| `wave` | Right shoulder raises and extends |
| `srk_pose` | Both arms fully horizontal to sides (DDLJ pose) |

---

## Project structure

```
robomind/
├── robot_env.py        # MuJoCo G1 wrapper — state extraction
├── llm_brain.py        # Nebius Token Factory API client
├── action_executor.py  # Maps action names → motor commands
├── world_model.py      # Parallel rollout predictor
├── demo.py             # Main demo loop with viewer
└── requirements.txt
```

---

## Research background

This system is an implementation of the pattern from:

- **SayCan** (Google, 2022) — LLM as high-level planner, pretrained skills for execution. [arXiv:2204.01691](https://arxiv.org/abs/2204.01691)
- **Inner Monologue** (Google, 2022) — Feedback loop: robot reports outcome, LLM re-plans. [arXiv:2207.05608](https://arxiv.org/abs/2207.05608)

Key insight: LLMs can't output motor commands, but they're excellent high-level reasoners. MuJoCo handles what it's good at — accurate physics prediction.

---

## Built with

- [MuJoCo](https://mujoco.org/) + [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Unitree G1 model)
- [Nebius Token Factory](https://tokenfactory.nebius.com/) — Qwen3-235B-A22B
- Python 3.13
