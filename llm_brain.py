import json
import os
from openai import OpenAI


class RobotBrain:
    def __init__(self, api_key=None):
        self.client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=api_key or os.environ["NEBIUS_API_KEY"],
        )
        self.model = "Qwen/Qwen3-235B-A22B-Instruct-2507"

    def think(
        self,
        state_description: dict,
        user_instruction: str,
        action_descriptions: dict | None = None,
    ) -> dict:
        """LLM reasons about the robot's state and decides what to do.

        Args:
            action_descriptions: pass executor.action_descriptions so the prompt
                                 always reflects the real action space.
        """
        if action_descriptions:
            action_lines = "\n".join(
                f"  {name}: {desc}" for name, desc in action_descriptions.items()
            )
            actions_block = f"Available actions:\n{action_lines}"
        else:
            actions_block = "Available actions: walk_forward, walk_backward, turn_left, turn_right, stop, stand_up, wave, srk_pose"

        prompt = f"""You are controlling a Unitree G1 humanoid robot in simulation.

CURRENT ROBOT STATE:
{json.dumps(state_description, indent=2)}

USER INSTRUCTION: {user_instruction}

{actions_block}

Respond in this exact JSON format (no markdown, no extra text):
{{
  "observation": "What you see in the robot's current state (1 sentence)",
  "reasoning": "Your analysis of what to do next (2-3 sentences)",
  "prediction": "What you predict will happen when this action executes (1 sentence)",
  "action": "one of the available actions",
  "confidence": 0.0
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a robot controller. Output only valid JSON with no markdown fences.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )

        content = response.choices[0].message.content.strip()
        # Strip markdown fences if model adds them anyway
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)


if __name__ == "__main__":
    brain = RobotBrain()
    fake_state = {
        "position": {"x": 0.0, "y": 0.0, "z": 0.78},
        "height_meters": 0.78,
        "forward_velocity_ms": 0.0,
        "roll_degrees": 1.2,
        "pitch_degrees": 0.5,
        "is_standing": True,
        "is_moving": False,
        "is_stable": True,
    }
    decision = brain.think(fake_state, "Walk forward toward the goal")
    print(json.dumps(decision, indent=2))
