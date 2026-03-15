import mujoco
import numpy as np


class SimWorldModel:
    """Uses MuJoCo as a world model by running parallel predictions."""

    STANDING_HEIGHT = 0.75
    HEIGHT_TOLERANCE = 0.30

    def __init__(self, model_path: str):
        self.model = mujoco.MjModel.from_xml_path(model_path)

    def predict_futures(
        self,
        current_state: dict,
        candidate_actions: dict,
        horizon: int = 60,
        num_variations: int = 5,
    ) -> dict:
        """
        For each candidate action, run `num_variations` noisy rollouts
        over `horizon` steps from the current state.

        Args:
            current_state: {"qpos": np.ndarray, "qvel": np.ndarray}
            candidate_actions: {action_name: np.ndarray of ctrl values}
            horizon: number of sim steps per rollout
            num_variations: number of noisy copies per action

        Returns:
            {action_name: {"trajectories", "stability_rate", "avg_forward_progress", "risk"}}
        """
        predictions = {}

        for action_name, action_values in candidate_actions.items():
            trajectories = []

            for v in range(num_variations):
                data = mujoco.MjData(self.model)
                data.qpos[:] = current_state["qpos"]
                data.qvel[:] = current_state["qvel"]

                # Increasing noise for each variation to model uncertainty
                noise_scale = 0.01 * v
                data.qvel[:] += np.random.normal(0, noise_scale, data.qvel.shape)

                traj = []
                for _ in range(horizon):
                    ctrl = action_values + np.random.normal(0, 0.005, action_values.shape)
                    np.copyto(data.ctrl, ctrl)
                    mujoco.mj_step(self.model, data)
                    height = float(data.qpos[2])
                    traj.append(
                        {
                            "position": data.qpos[:3].copy(),
                            "height": height,
                            "velocity": data.qvel[:3].copy(),
                            "stable": abs(height - self.STANDING_HEIGHT) < self.HEIGHT_TOLERANCE,
                        }
                    )
                trajectories.append(traj)

            # Aggregate metrics
            stability_rate = float(
                np.mean([all(s["stable"] for s in traj) for traj in trajectories])
            )
            avg_forward_progress = float(
                np.mean(
                    [traj[-1]["position"][0] - traj[0]["position"][0] for traj in trajectories]
                )
            )

            if stability_rate > 0.8:
                risk = "LOW"
            elif stability_rate > 0.5:
                risk = "MEDIUM"
            else:
                risk = "HIGH"

            predictions[action_name] = {
                "trajectories": trajectories,
                "stability_rate": round(stability_rate, 2),
                "avg_forward_progress": round(avg_forward_progress, 3),
                "risk": risk,
            }

        return predictions
