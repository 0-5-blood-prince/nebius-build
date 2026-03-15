import mujoco
import numpy as np
import os


class ActionExecutor:
    """Maps high-level LLM action names to low-level motor commands."""

    # How many sim steps per high-level action
    DEFAULT_STEPS = 150

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.nu = model.nu

        if os.environ.get("ROBOMIND_DEBUG"):
            print(f"[ActionExecutor] {self.nu} actuators:")
            for i in range(self.nu):
                print(f"  [{i}] {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")

        self.actions = {
            "stand_up":      self._stand(),
            "stop":          self._stand(),
            "walk_forward":  self._walk(0.4),
            "walk_backward": self._walk(-0.3),
            "turn_left":     self._turn(0.3),
            "turn_right":    self._turn(-0.3),
            "wave":          self._wave(),
            "srk_pose":      self._srk_pose(),
        }

        # Human-readable descriptions — used to build the LLM prompt dynamically.
        # Update this whenever you add a new action.
        self.action_descriptions = {
            "stand_up":      "Stand upright in a neutral position",
            "stop":          "Stop all movement and hold current position",
            "walk_forward":  "Walk forward",
            "walk_backward": "Walk backward",
            "turn_left":     "Turn left in place",
            "turn_right":    "Turn right in place",
            "wave":          "Raise right arm and wave",
            "srk_pose":      "Shah Rukh Khan DDLJ pose — both arms extended fully horizontal to the sides with a slight backward lean",
        }

    def _stand(self):
        return np.zeros(self.nu)

    def _walk(self, speed: float):
        """Drive hip pitch and knee joints for a basic walk motion.
        Indices from real G1 scene.xml actuator list:
          0=left_hip_pitch, 3=left_knee, 6=right_hip_pitch, 9=right_knee
        """
        a = np.zeros(self.nu)
        a[0] = speed * 0.5   # left_hip_pitch_joint
        a[3] = -speed * 0.3  # left_knee_joint (flex)
        a[6] = speed * 0.5   # right_hip_pitch_joint
        a[9] = -speed * 0.3  # right_knee_joint (flex)
        return a

    def _turn(self, yaw_rate: float):
        """Hip yaw joints: 2=left_hip_yaw, 8=right_hip_yaw."""
        a = np.zeros(self.nu)
        a[2] = yaw_rate    # left_hip_yaw_joint
        a[8] = -yaw_rate   # right_hip_yaw_joint
        return a

    def _wave(self):
        """Right shoulder pitch (22) and roll (23) for a visible wave."""
        a = np.zeros(self.nu)
        a[22] = 1.2   # right_shoulder_pitch_joint — raise arm
        a[23] = 0.5   # right_shoulder_roll_joint  — extend out
        return a

    def _srk_pose(self):
        """Shah Rukh Khan iconic DDLJ arms-spread pose.
        Both arms fully horizontal to the sides (T-pose), chest open, slight backward lean.

        Joint ranges (from g1.xml):
          left_shoulder_roll:  -1.59 to +2.25  (+ve = arm raises outward LEFT)
          right_shoulder_roll: -2.25 to +1.59  (-ve = arm raises outward RIGHT)
          left_elbow:          -1.05 to +2.09  (0 = straight)
          right_elbow:         -1.05 to +2.09  (0 = straight)
          waist_pitch:         -0.52 to +0.52  (-ve = lean back)

        Indices: 14=waist_pitch, 15=L_shoulder_pitch, 16=L_shoulder_roll,
                 17=L_shoulder_yaw, 18=L_elbow,
                 22=R_shoulder_pitch, 23=R_shoulder_roll,
                 24=R_shoulder_yaw, 25=R_elbow
        """
        a = np.zeros(self.nu)
        a[14] = -0.08  # waist_pitch          — very subtle lean back (safer CoM)
        a[15] =  0.0   # left_shoulder_pitch  — neutral (no forward/back swing)
        a[16] =  1.55  # left_shoulder_roll   — raise left arm fully out to side
        a[17] =  0.0   # left_shoulder_yaw    — palm faces down
        a[18] =  0.0   # left_elbow           — straight arm
        a[22] =  0.0   # right_shoulder_pitch — neutral
        a[23] = -1.55  # right_shoulder_roll  — raise right arm fully out to side
        a[24] =  0.0   # right_shoulder_yaw   — palm faces down
        a[25] =  0.0   # right_elbow          — straight arm
        return a

    def execute(self, action_name: str, steps: int = DEFAULT_STEPS) -> list:
        """Execute action for N steps; return position trajectory."""
        cmd = self.actions.get(action_name, self._stand())
        trajectory = []
        for _ in range(steps):
            np.copyto(self.data.ctrl, cmd)
            mujoco.mj_step(self.model, self.data)
            trajectory.append(self.data.qpos[:3].copy())
        return trajectory
