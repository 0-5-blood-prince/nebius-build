"""
RoboMind — LLM-Orchestrated Robot Control with Predictive Planning

Usage:
  python demo.py           # uses NEBIUS_API_KEY env var
  python demo.py --stub    # runs with pre-scripted decisions (no API key needed)
"""
import json
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from action_executor import ActionExecutor
from llm_brain import RobotBrain
from robot_env import G1Environment, MODEL_PATH
from world_model import SimWorldModel

SEPARATOR = "=" * 65

DEMO_INSTRUCTIONS = [
    "Walk forward to explore the area",
    "Stop and look around — are you stable?",
    "Do the Shah Rukh Khan pose — arms wide open like in DDLJ",
    "Return to standing position",
    "Wave to the audience and then stand still",
]

# Pre-scripted decisions used when --stub flag is passed (no API key needed)
STUB_DECISIONS = [
    {
        "observation": "Robot is standing upright and stationary at the origin.",
        "reasoning": "User wants forward exploration. Robot is stable with no tilt. Walk forward is safe.",
        "prediction": "Robot will move forward ~0.03 m per step while remaining stable at ~0.79 m height.",
        "action": "walk_forward",
        "confidence": 0.92,
    },
    {
        "observation": "Robot has moved forward and is still stable.",
        "reasoning": "Instruction is to stop and assess. Robot is upright. Issuing stop to hold position.",
        "prediction": "Robot will decelerate and hold its current position without falling.",
        "action": "stop",
        "confidence": 0.97,
    },
    {
        "observation": "Robot is stationary and stable. SRK pose requested.",
        "reasoning": "Shah Rukh Khan's iconic DDLJ pose requires both arms extended wide to the sides with a slight backward lean. Robot is balanced so this is safe to execute.",
        "prediction": "Both arms will extend outward to form a cross shape — the classic SRK silhouette.",
        "action": "srk_pose",
        "confidence": 0.96,
    },
    {
        "observation": "SRK pose complete. Arms need to return to neutral before next action.",
        "reasoning": "Resetting to standing position to ensure stability before waving.",
        "prediction": "Arms will lower and robot will return to neutral upright stance.",
        "action": "stand_up",
        "confidence": 0.99,
    },
    {
        "observation": "Robot is standing upright and stable.",
        "reasoning": "Audience greeting requested. Robot is balanced — safe to raise arm for wave.",
        "prediction": "Right arm will rise and wave before returning to rest.",
        "action": "wave",
        "confidence": 0.95,
    },
]


def _sync_viewer(viewer, env, steps):
    """Step simulation and keep viewer in sync."""
    for _ in range(steps):
        mujoco.mj_step(env.model, env.data)
        viewer.sync()


def run_demo(stub: bool = False):
    print(SEPARATOR)
    print("  RoboMind: LLM-Orchestrated Robot Control")
    if stub:
        print("  [STUB MODE — no API key required]")
    print(SEPARATOR)

    print("\nInitializing environment...")
    env = G1Environment()
    executor = ActionExecutor(env.model, env.data)
    world_model = SimWorldModel(MODEL_PATH)
    brain = None if stub else RobotBrain()
    print("Ready. Opening MuJoCo viewer...\n")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Frame the full robot: pull back, look at mid-torso height
        viewer.cam.lookat[:] = [0.0, 0.0, 0.85]  # look at torso center
        viewer.cam.distance = 3.5                  # far enough to see head+feet
        viewer.cam.elevation = -15                 # slight downward angle
        viewer.cam.azimuth = 90                    # front-facing view

        # Let physics settle before starting
        for _ in range(200):
            mujoco.mj_step(env.model, env.data)
            viewer.sync()

        for i, instruction in enumerate(DEMO_INSTRUCTIONS):
            if not viewer.is_running():
                break

            print(f"\n{SEPARATOR}")
            print(f"  Step {i + 1}/{len(DEMO_INSTRUCTIONS)}")
            print(f"  USER: {instruction}")
            print(SEPARATOR)

            # 1. Perceive
            state = env.get_state_description()
            print(f"\n[PERCEPTION]\n{json.dumps(state, indent=2)}")

            # 2. Reason
            if stub:
                decision = STUB_DECISIONS[i]
                print("\n[LLM REASONING] (stub)")
            else:
                print("\n[LLM REASONING] Calling Nebius Token Factory...")
                decision = brain.think(state, instruction, executor.action_descriptions)

            print(f"  Observation : {decision['observation']}")
            print(f"  Reasoning   : {decision['reasoning']}")
            print(f"  Action      : {decision['action']}  (confidence: {decision['confidence']})")
            print(f"  Prediction  : {decision['prediction']}")

            # 3. World-model prediction before execution
            raw_state = env.get_raw_state()
            action_cmd = executor.actions.get(decision["action"], np.zeros(env.model.nu))
            print("\n[WORLD MODEL] Running parallel rollouts...")
            predictions = world_model.predict_futures(
                raw_state, {decision["action"]: action_cmd}
            )
            pred = predictions[decision["action"]]
            print(f"  Stability rate     : {pred['stability_rate'] * 100:.0f}%")
            print(f"  Avg forward travel : {pred['avg_forward_progress']:.3f} m")
            print(f"  Risk level         : {pred['risk']}")

            # 4. Safety gate
            if pred["risk"] == "HIGH" and float(decision["confidence"]) < 0.7:
                print("\n[SAFETY] High risk detected — re-planning...")
                if stub:
                    decision["action"] = "stop"
                    print("  New action: stop (stub fallback)")
                else:
                    augmented = {**state, "warning": "World model predicts HIGH risk"}
                    decision = brain.think(
                        augmented,
                        instruction + " (CAUTION: previous plan predicted high fall risk)",
                        executor.action_descriptions,
                    )
                    print(f"  New action: {decision['action']}")
                action_cmd = executor.actions.get(decision["action"], np.zeros(env.model.nu))

            # 5. Execute — step sim and keep viewer live
            print(f"\n[EXECUTION] Running: {decision['action']} ...")
            for _ in range(150):
                if not viewer.is_running():
                    break
                np.copyto(env.data.ctrl, action_cmd)
                mujoco.mj_step(env.model, env.data)
                viewer.sync()

            # 6. Observe outcome
            new_state = env.get_state_description()
            height_delta = new_state["height_meters"] - state["height_meters"]
            fwd_delta = new_state["position"]["x"] - state["position"]["x"]
            print("[OUTCOME]")
            print(f"  Height change   : {height_delta:+.3f} m")
            print(f"  Forward travel  : {fwd_delta:+.3f} m")
            print(f"  Still stable    : {new_state['is_stable']}")

            # Pause between steps so viewer and narration can breathe
            pause_start = time.time()
            while time.time() - pause_start < 2.0 and viewer.is_running():
                mujoco.mj_step(env.model, env.data)
                viewer.sync()

    print(f"\n{SEPARATOR}")
    print("  Demo complete.")
    print(SEPARATOR)


if __name__ == "__main__":
    stub_mode = "--stub" in sys.argv or not os.environ.get("NEBIUS_API_KEY")
    if stub_mode and "--stub" not in sys.argv:
        print("Note: NEBIUS_API_KEY not set — running in stub mode. Pass --stub to suppress this.")
    run_demo(stub=stub_mode)
