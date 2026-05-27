import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from CSCG_helpers import Plotting, Reasoning
GRANULAR_ROOM = np.array(
    [
        [4, 2, 3, 0, 3, 4, 4, 4],
        [4, 4, 3, 2, 3, 2, 3, 4],
        [4, 4, 2, 0, 4, 2, 4, 0],
        [0, 2, 4, 4, 3, 0, 0, 2],
        [3, 3, 4, 0, 4, 1, 3, 0],
        [2, 4, 2, 3, 3, 3, 2, 0],
    ]
)


class Experiment:
    """Small experiment wrapper around the existing CSCG/STP helpers."""

    def __init__(
        self,
        name,
        room=None,
        model=None,
        graph=None,
        T=None,
        observations=None,
        actions=None,
        starts=None,
        targets=None,
        plan_method=Reasoning.STP,
        length=5000,
        clone_count=25,
        seed=42,
        retrain_model=False,
        wavefront_steps=10,
        planning_steps=10,
        experiments_dir="experiments",
        models_dir="models",
    ):
        self.name = name
        self.room = np.asarray(room if room is not None else GRANULAR_ROOM)
        self.model = model
        self.graph = graph
        self.T = None if T is None else np.asarray(T, dtype=float)
        self.observations = None if observations is None else np.asarray(observations, dtype=np.int64)
        self.actions = None if actions is None else np.asarray(actions, dtype=np.int64)
        self.starts = self._as_list(starts)
        self.targets = self._as_list(targets)
        self.plan_method = plan_method
        self.length = length
        self.clone_count = clone_count
        self.seed = seed
        self.retrain_model = retrain_model
        self.wavefront_steps = wavefront_steps
        self.planning_steps = planning_steps

        self.experiments_dir = Path(experiments_dir)
        self.models_dir = Path(models_dir)
        self.path = self.experiments_dir / self.name
        self.model_path = self.models_dir / f"{self.name}.pkl"

        self.transition_weights = None
        self.wavefront_values = []
        self.planning_values = []
        self.action_plan = []
        self.state_plan = []
        self.obs_plan = []
        self.chosen_actions = []
        self.notes = []

    def run(self, stages=None):
        """Run selected pipeline stages and save the resulting artifacts."""
        stages = set(stages or ("source", "wavefront", "planning"))
        self.path.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if "source" in stages:
            self.build_transition()
        elif self.T is None:
            self.load_artifacts()

        if "wavefront" in stages and self.targets:
            self.run_wavefront()

        if "planning" in stages and self.starts:
            self.run_planning()

        self.save_artifacts()
        return self

    def build_transition(self):
        """Resolve room/model/graph/T input into the action transition tensor T."""
        if self.T is not None:
            self.transition_weights = self.T.copy()
            return self.transition_weights

        if self.model is None and self.graph is None:
            from visualize_stp import setup_navigation_model

            self.model, self.observations, self.actions, _rc, self.room, _cmap = setup_navigation_model(
                selected_room=self.room,
                name=self.name,
                retrain_models=self.retrain_model,
                length=self.length,
                clone_count=self.clone_count,
                seed=self.seed,
            )

        if self.model is not None:
            self.T = np.asarray(getattr(self.model, "T", getattr(self.model, "C", None)), dtype=float)
            if self.T is None:
                raise ValueError("Provided model does not expose T or C.")
            if not self.model_path.exists():
                self._save_model()
        else:
            self.T = self.graph_to_T(self.graph)

        self.transition_weights = self.T.copy()
        return self.transition_weights

    def run_wavefront(self, steps=None, targets=None):
        """Run the target-to-start STP wavefront and keep each activity frame."""
        self._require_T()
        targets = self._as_list(targets) or self.targets
        steps = self.wavefront_steps if steps is None else steps
        values = self._one_hot(targets)
        transition_weights = self.T.copy()
        self.wavefront_values = [values.copy()]

        for _ in range(steps):
            values, transition_weights = self.plan_method(values, transition_weights)
            self.wavefront_values.append(values.copy())

        self.transition_weights = transition_weights
        return transition_weights

    def run_planning(self, steps=None, starts=None):
        """Propagate from starts through T' and derive a concrete action plan."""
        self._require_T()
        starts = self._as_list(starts) or self.starts
        steps = self.planning_steps if steps is None else steps
        transition_weights = self.transition_weights if self.transition_weights is not None else self.T
        initial_values = self._one_hot(starts)
        values = initial_values.copy()
        self.planning_values = [values.copy()]
        self.chosen_actions = [int(Reasoning.select_action(values, transition_weights))]

        for _ in range(steps):
            values = Reasoning.propogate(values, transition_weights, initial_values)
            self.planning_values.append(values.copy())
            self.chosen_actions.append(int(Reasoning.select_action(values, transition_weights)))

        if self.model is not None and starts:
            self.state_plan, self.obs_plan, self.action_plan = Reasoning.plan_path(
                starts[0],
                transition_weights,
                self.model.n_clones,
            )
            self.action_plan = [int(action) for action in self.action_plan]
            self.state_plan = [int(state) for state in self.state_plan]
            self.obs_plan = [None if obs is None else int(obs) for obs in self.obs_plan]

        return self.action_plan or self.chosen_actions

    def visualize(self, mode="combined", interactive=True):
        """Use the existing visualization helpers for CSCG-backed experiments."""
        if not interactive:
            return self.save_visualizations()

        self._require_model_context()
        image_path = str(self.path / f"{self.name}-{mode}.png")

        if mode == "wavefront":
            from visualize_stp import plot_reasoning

            return plot_reasoning(
                self.targets,
                model=self.model,
                observations=self.observations,
                actions=self.actions,
                image_path=image_path,
            )
        if mode == "planning":
            from visualize_stp import plot_planning

            transition_weights = self.transition_weights if self.transition_weights is not None else self.T
            return plot_planning(
                self.starts,
                transition_weights,
                model=self.model,
                observations=self.observations,
                actions=self.actions,
                image_path=image_path,
            )
        if mode == "path":
            from visualize_stp import show_graph_and_plan

            transition_weights = self.transition_weights if self.transition_weights is not None else self.T
            return show_graph_and_plan(self.starts, transition_weights, name=self.name)

        from visualize_stp import plot_reasoning_then_planning

        return plot_reasoning_then_planning(
            self.targets,
            self.starts,
            model=self.model,
            observations=self.observations,
            actions=self.actions,
            image_path=image_path,
        )

    def save_visualizations(self):
        """Write graph heatmaps for saved wavefront/planning frames."""
        self._require_model_context()
        written = []
        transition_weights = self.transition_weights if self.transition_weights is not None else self.T

        for prefix, frames in (("wavefront", self.wavefront_values), ("planning", self.planning_values)):
            for index, values in enumerate(frames):
                output_file = self.path / f"{prefix}-{index:03d}.png"
                Plotting.plot_heat_map(
                    self.model,
                    self.observations,
                    self.actions,
                    values,
                    output_file=str(output_file),
                    transition_weights=transition_weights,
                    edge_label_mode="int",
                    vertex_label_mode="value",
                )
                written.append(str(output_file))

        return written

    def add_note(self, txt):
        self.notes.append({"created_at": datetime.now().isoformat(timespec="seconds"), "text": txt})
        self.save_artifacts()

    def save_artifacts(self):
        self.path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "name": self.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_path": str(self.model_path) if self.model is not None else None,
            "starts": self.starts,
            "targets": self.targets,
            "length": self.length,
            "clone_count": self.clone_count,
            "seed": self.seed,
            "wavefront_steps": self.wavefront_steps,
            "planning_steps": self.planning_steps,
            "chosen_actions": self.chosen_actions,
            "action_plan": self.action_plan,
            "state_plan": self.state_plan,
            "obs_plan": self.obs_plan,
            "notes": self.notes,
        }
        with open(self.path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        arrays = {"room": self.room}
        if self.T is not None:
            arrays["T"] = self.T
        if self.transition_weights is not None:
            arrays["transition_weights"] = self.transition_weights
        if self.observations is not None:
            arrays["observations"] = self.observations
        if self.actions is not None:
            arrays["actions"] = self.actions
        if self.wavefront_values:
            arrays["wavefront_values"] = np.asarray(self.wavefront_values)
        if self.planning_values:
            arrays["planning_values"] = np.asarray(self.planning_values)
        np.savez_compressed(self.path / "arrays.npz", **arrays)

    def load_artifacts(self):
        arrays_file = self.path / "arrays.npz"
        if not arrays_file.exists():
            raise FileNotFoundError(f"No saved arrays found for experiment '{self.name}'.")

        arrays = np.load(arrays_file, allow_pickle=True)
        self.room = arrays["room"]
        self.T = arrays["T"] if "T" in arrays else None
        self.transition_weights = arrays["transition_weights"] if "transition_weights" in arrays else self.T
        self.observations = arrays["observations"] if "observations" in arrays else self.observations
        self.actions = arrays["actions"] if "actions" in arrays else self.actions
        self.wavefront_values = list(arrays["wavefront_values"]) if "wavefront_values" in arrays else []
        self.planning_values = list(arrays["planning_values"]) if "planning_values" in arrays else []

        metadata_file = self.path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.starts = metadata.get("starts", self.starts)
            self.targets = metadata.get("targets", self.targets)
            self.chosen_actions = metadata.get("chosen_actions", [])
            self.action_plan = metadata.get("action_plan", [])
            self.state_plan = metadata.get("state_plan", [])
            self.obs_plan = metadata.get("obs_plan", [])
            self.notes = metadata.get("notes", [])

        if self.model is None and self.model_path.exists():
            with open(self.model_path, "rb") as f:
                loaded = pickle.load(f)
            self.model = loaded[0] if isinstance(loaded, tuple) else loaded

        return self

    @staticmethod
    def list_experiments(experiments_dir="experiments"):
        root = Path(experiments_dir)
        if not root.exists():
            return []
        return sorted(path.name for path in root.iterdir() if path.is_dir())

    @staticmethod
    def exists(name, experiments_dir="experiments"):
        return (Path(experiments_dir) / name).is_dir()

    @staticmethod
    def graph_to_T(graph, n_actions=4):
        """Convert common graph/adjacency formats into an action tensor."""
        if isinstance(graph, dict):
            n_states = graph.get("n_states")
            edges = graph.get("edges", [])
            if n_states is None:
                n_states = 1 + max(max(src, dst) for src, dst, *_rest in edges)
            T = np.zeros((n_actions, n_states, n_states), dtype=float)
            for edge in edges:
                src, dst = edge[:2]
                action = edge[2] if len(edge) > 2 else 0
                weight = edge[3] if len(edge) > 3 else 1.0
                T[int(action), int(src), int(dst)] = float(weight)
            return Experiment._normalize_T(T)

        graph = np.asarray(graph, dtype=float)
        if graph.ndim == 2:
            T = np.zeros((n_actions, graph.shape[0], graph.shape[1]), dtype=float)
            T[0] = graph
            return Experiment._normalize_T(T)
        if graph.ndim == 3:
            return Experiment._normalize_T(graph)
        raise ValueError("Graph must be an adjacency matrix, an action tensor, or an edge-list dict.")

    @staticmethod
    def _normalize_T(T):
        T = np.asarray(T, dtype=float).copy()
        norm = T.sum(axis=2, keepdims=True)
        norm[norm == 0] = 1
        return T / norm

    @staticmethod
    def _as_list(values):
        if values is None:
            return []
        if isinstance(values, (int, np.integer)):
            return [int(values)]
        return [int(value) for value in values]

    def _one_hot(self, nodes):
        if self.model is not None:
            size = int(np.sum(self.model.n_clones))
        else:
            self._require_T()
            size = self.T.shape[1]
        values = np.zeros(size)
        for node in nodes:
            values[int(node)] = 1.0
        return values

    def _require_T(self):
        if self.T is None:
            self.build_transition()
        if self.T is None:
            raise ValueError("Experiment has no transition tensor. Provide room, model, graph, or T.")

    def _require_model_context(self):
        self._require_T()
        if self.model is None or self.observations is None or self.actions is None:
            raise ValueError("Interactive CSCG visualizations require a model, observations, and actions.")

    def _save_model(self):
        if self.model is None:
            return
        self.models_dir.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump((self.model, []), f, protocol=5)


def comparison(experiments):
    rows = []
    for exp in experiments:
        if not isinstance(exp, Experiment):
            exp = Experiment(exp).load_artifacts()
        rows.append(
            {
                "name": exp.name,
                "starts": exp.starts,
                "targets": exp.targets,
                "chosen_actions": exp.chosen_actions,
                "action_plan": exp.action_plan,
            }
        )
    return rows
