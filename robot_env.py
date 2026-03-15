import mujoco
import numpy as np
import json

MODEL_PATH = "mujoco_menagerie/unitree_g1/scene.xml"


class G1Environment:
    def __init__(self, model_path=MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        # Reset to default standing pose
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def get_state_description(self):
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        vel = self.data.qvel[:3]

        height = float(pos[2])
        forward_vel = float(vel[0])

        # Roll/pitch from quaternion (w, x, y, z order in MuJoCo)
        w, x, y, z = quat
        roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2)))
        pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))

        return {
            "position": {
                "x": round(float(pos[0]), 2),
                "y": round(float(pos[1]), 2),
                "z": round(float(pos[2]), 2),
            },
            "height_meters": round(height, 2),
            "forward_velocity_ms": round(forward_vel, 2),
            "roll_degrees": round(np.degrees(roll), 1),
            "pitch_degrees": round(np.degrees(pitch), 1),
            "is_standing": bool(height > 0.6),
            "is_moving": bool(abs(forward_vel) > 0.1),
            "is_stable": bool(abs(np.degrees(roll)) < 15 and abs(np.degrees(pitch)) < 15),
        }

    def get_raw_state(self):
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }

    def step(self, action=None):
        if action is not None:
            np.copyto(self.data.ctrl, action)
        mujoco.mj_step(self.model, self.data)
        return self.get_state_description()


if __name__ == "__main__":
    env = G1Environment()
    state = env.get_state_description()
    print("G1 state:")
    print(json.dumps(state, indent=2))
    print("\nG1 loaded successfully!")
