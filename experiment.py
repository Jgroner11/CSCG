import json
import pickle
import re
import types
from datetime import datetime
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from CSCG_helpers import Plotting, Reasoning


DEFAULT_N_OBS = 4


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
        plan_method=Reasoning.STP1,
        seq_length=5000,
        n_clones=25,
        seed=42,
        retrain_model=False,
        wavefront_steps=10,
        planning_steps=10,
        graph_normalize=False,
        experiments_dir="experiments",
        models_dir="models",
    ):
        """Create an experiment configuration.

        Parameters
        ----------
        name : str
            Experiment name. Artifacts are saved under ``experiments_dir/name``.
        room : array-like, optional
            CSCG room layout used to train/load a navigation model when no model,
            graph, or transition tensor is provided.
        model : object, optional
            Prebuilt model exposing ``T`` or ``C`` as its transition tensor.
        graph : str, optional
            Name of a CS Academy graph text file in ``graphs/``. ``"graph1"``
            and ``"graph1.txt"`` both resolve to ``graphs/graph1.txt``.
            Graph text can be copied from https://csacademy.com/app/graph_editor/.
        T : array-like, optional
            Prebuilt transition tensor with shape ``(actions, states, states)``.
        observations : array-like, optional
            Observation sequence used for CSCG-backed visualization context.
        actions : array-like, optional
            Action sequence used for CSCG-backed visualization context.
        starts : int or iterable of int, optional
            Start state ids for planning.
        targets : int or iterable of int, optional
            Target state ids for wavefront propagation.
        plan_method : callable
            Planning update function, such as ``Reasoning.STP1``.
        seq_length : int
            Sequence length used when building a CSCG model from a room.
        n_clones : int
            Clone count used when building a CSCG model from a room.
        seed : int
            Random seed used when building a CSCG model from a room.
        retrain_model : bool
            Whether to retrain the CSCG model when building from a room.
        wavefront_steps : int
            Default number of wavefront propagation steps.
        planning_steps : int
            Default number of planning propagation steps.
        graph_normalize : bool
            Whether to normalize outgoing edge weights per source node so they sum
            to 1, making T a valid probability transition matrix. When False and no
            edge weights are specified, each existing edge has value 1.
        experiments_dir : str or Path
            Directory for saved experiment artifacts.
        models_dir : str or Path
            Directory for saved model artifacts.
        """
        self.name = name
        self.room = None if room is None else np.asarray(room)
        self.model = model
        self.graph = graph
        self.T = None if T is None else np.asarray(T, dtype=float)
        self.observations = None if observations is None else np.asarray(observations, dtype=np.int64)
        self.actions = None if actions is None else np.asarray(actions, dtype=np.int64)
        self.starts = self._as_list(starts)
        self.targets = self._as_list(targets)
        self.plan_method = plan_method
        self.seq_length = seq_length
        self.n_clones = n_clones
        self.seed = seed
        self.retrain_model = retrain_model
        self.wavefront_steps = wavefront_steps
        self.planning_steps = planning_steps
        self.graph_normalize = graph_normalize

        self.experiments_dir = Path(experiments_dir)
        self.models_dir = Path(models_dir)
        self.path = self.experiments_dir / self.name
        self.model_path = self.models_dir / f"{self.name}.pkl"

        self.transition_weights = None
        self.decoded_states = None
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
            if self.room is None:
                raise ValueError(
                    "Experiment needs a source: provide room, model, graph, or T. "
                    "Use Experiment.get(name) to load an existing experiment."
                )
            from visualize_stp import setup_navigation_model

            self.model, self.observations, self.actions, _rc, self.room, _cmap = setup_navigation_model(
                selected_room=self.room,
                name=self.name,
                retrain_models=self.retrain_model,
                seq_length=self.seq_length,
                n_clones=self.n_clones,
                seed=self.seed,
            )

        if self.model is not None:
            self.T = np.asarray(getattr(self.model, "T", getattr(self.model, "C", None)), dtype=float)
            if self.T is None:
                raise ValueError("Provided model does not expose T or C.")
            if not self.model_path.exists():
                self._save_model()
        else:
            self.T = self.graph_to_T(
                self.graph,
                normalize=self.graph_normalize,
            )
            self._set_graph_model_context()

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

    def visualize(self, mode="combined", interactive=True, starts=None, targets=None, per_action=False, plan_method=None):
        """Use the existing visualization helpers for CSCG-backed experiments."""
        if not interactive:
            return self.save_visualizations()

        self._require_model_context()
        self._ensure_decoded_states(save=True)
        starts = self._as_list(starts) or self.starts
        targets = self._as_list(targets) or self.targets
        print(f"targets: {targets}")
        print(f"starts:  {starts}")
        image_path = str(self.path / f"{self.name}-{mode}.png")

        if mode == "wavefront":
            from visualize_stp import plot_reasoning

            return plot_reasoning(
                targets,
                model=self.model,
                observations=self.observations,
                actions=self.actions,
                decoded_states=self.decoded_states,
                image_path=image_path,
            )
        if mode == "planning":
            from visualize_stp import plot_planning

            transition_weights = self.transition_weights if self.transition_weights is not None else self.T
            return plot_planning(
                starts,
                transition_weights,
                model=self.model,
                observations=self.observations,
                actions=self.actions,
                decoded_states=self.decoded_states,
                image_path=image_path,
            )
        if mode == "path":
            from visualize_stp import show_graph_and_plan

            transition_weights = self.transition_weights if self.transition_weights is not None else self.T
            return show_graph_and_plan(starts, transition_weights, name=self.name)

        if per_action:
            from visualize_stp import plot_reasoning_then_planning_per_action

            return plot_reasoning_then_planning_per_action(
                targets,
                starts,
                model=self.model,
                observations=self.observations,
                actions=self.actions,
                decoded_states=self.decoded_states,
                image_path=image_path,
            )

        from visualize_stp import plot_reasoning_then_planning

        return plot_reasoning_then_planning(
            targets,
            starts,
            model=self.model,
            observations=self.observations,
            actions=self.actions,
            decoded_states=self.decoded_states,
            image_path=image_path,
            plan_method=plan_method if plan_method is not None else self.plan_method,
        )

    def save_visualizations(self):
        """Write graph heatmaps for saved wavefront/planning frames."""
        self._require_model_context()
        self._ensure_decoded_states(save=True)
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
                    states=self.decoded_states,
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
            "seq_length": self.seq_length,
            "n_clones": self.n_clones,
            "seed": self.seed,
            "wavefront_steps": self.wavefront_steps,
            "planning_steps": self.planning_steps,
            "graph": self.graph,
            "graph_normalize": self.graph_normalize,
            "chosen_actions": self.chosen_actions,
            "action_plan": self.action_plan,
            "state_plan": self.state_plan,
            "obs_plan": self.obs_plan,
            "notes": self.notes,
        }
        with open(self.path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        arrays = {}
        if self.room is not None:
            arrays["room"] = self.room
        if self.T is not None:
            arrays["T"] = self.T
        if self.transition_weights is not None:
            arrays["transition_weights"] = self.transition_weights
        if self.observations is not None:
            arrays["observations"] = self.observations
        if self.actions is not None:
            arrays["actions"] = self.actions
        if self.decoded_states is None and self.model is not None and self.observations is not None and self.actions is not None:
            self.decoded_states = self.model.decode(self.observations, self.actions)[1]
        if self.decoded_states is not None:
            arrays["decoded_states"] = self.decoded_states
        if self.wavefront_values:
            arrays["wavefront_values"] = np.asarray(self.wavefront_values)
        if self.planning_values:
            arrays["planning_values"] = np.asarray(self.planning_values)
        np.savez_compressed(self.path / "arrays.npz", **arrays)

    def _ensure_decoded_states(self, save=False):
        if self.decoded_states is not None:
            return
        if self.model is None or self.observations is None or self.actions is None:
            return
        self.decoded_states = self.model.decode(self.observations, self.actions)[1]
        if save:
            self.save_artifacts()

    def load_artifacts(self):
        arrays_file = self.path / "arrays.npz"
        if not arrays_file.exists():
            raise FileNotFoundError(f"No saved arrays found for experiment '{self.name}'.")

        arrays = np.load(arrays_file, allow_pickle=True)
        self.room = arrays["room"] if "room" in arrays else self.room
        self.T = arrays["T"] if "T" in arrays else None
        self.transition_weights = arrays["transition_weights"] if "transition_weights" in arrays else self.T
        self.observations = arrays["observations"] if "observations" in arrays else self.observations
        self.actions = arrays["actions"] if "actions" in arrays else self.actions
        self.decoded_states = arrays["decoded_states"] if "decoded_states" in arrays else self.decoded_states
        self.wavefront_values = list(arrays["wavefront_values"]) if "wavefront_values" in arrays else []
        self.planning_values = list(arrays["planning_values"]) if "planning_values" in arrays else []

        metadata_file = self.path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            model_path = metadata.get("model_path")
            if model_path is not None:
                self.model_path = Path(model_path)
            self.starts = metadata.get("starts", self.starts)
            self.targets = metadata.get("targets", self.targets)
            self.seq_length = metadata.get("seq_length", self.seq_length)
            self.n_clones = metadata.get("n_clones", self.n_clones)
            self.seed = metadata.get("seed", self.seed)
            self.wavefront_steps = metadata.get("wavefront_steps", self.wavefront_steps)
            self.planning_steps = metadata.get("planning_steps", self.planning_steps)
            self.graph = metadata.get("graph", self.graph)
            self.graph_normalize = metadata.get("graph_normalize", self.graph_normalize)
            self.chosen_actions = metadata.get("chosen_actions", [])
            self.action_plan = metadata.get("action_plan", [])
            self.state_plan = metadata.get("state_plan", [])
            self.obs_plan = metadata.get("obs_plan", [])
            self.notes = metadata.get("notes", [])

        if self.model is None and self.model_path.exists():
            with open(self.model_path, "rb") as f:
                loaded = pickle.load(f)
            self.model = loaded[0] if isinstance(loaded, tuple) else loaded
        elif self.model is None and self.graph is not None and self.T is not None:
            self._set_graph_model_context()

        return self

    @staticmethod
    def get(name, experiments_dir="experiments", models_dir="models"):
        exp = Experiment(name, experiments_dir=experiments_dir, models_dir=models_dir)
        return exp.load_artifacts()

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
    def delete_all_experiments(experiments_dir="experiments"):
        import shutil
        root = Path(experiments_dir)
        if not root.exists():
            return
        for path in root.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
        print(f"Deleted all experiments in '{root}'.")

    @staticmethod
    def comparison(experiments, image_dir="figures", flip=True, rotation=0.9, show=True):
        resolved = []
        for exp in experiments:
            if isinstance(exp, Experiment):
                resolved.append(exp)
            elif isinstance(exp, str):
                resolved.append(Experiment.get(exp))
            else:
                raise TypeError("experiments must contain Experiment objects or experiment name strings.")

        if not resolved:
            raise ValueError("comparison needs at least one experiment.")

        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)

        states = []
        for exp in resolved:
            exp._require_model_context()
            exp._ensure_decoded_states(save=True)
            if not exp.targets:
                raise ValueError(f"Experiment '{exp.name}' has no targets for wavefront visualization.")
            if not exp.starts:
                raise ValueError(f"Experiment '{exp.name}' has no starts for planning visualization.")

            wavefront_values = exp._one_hot(exp.targets)
            transition_weights = exp.T.copy()
            planning_initial = exp._one_hot(exp.starts)
            states.append({
                "exp": exp,
                "wavefront_values": wavefront_values,
                "transition_weights": transition_weights,
                "planning_initial": planning_initial,
                "planning_values": planning_initial.copy(),
                "step": 0,
                "image": image_dir / f"comparison-{Experiment._safe_filename(exp.name)}.png",
            })

        mode = "Wavefront"

        fig_width = max(7, 4.8 * len(resolved))
        fig, axes_2d = plt.subplots(1, len(resolved), figsize=(fig_width, 5), squeeze=False)
        axes = axes_2d[0]
        img_displays = []

        def render(i, state):
            exp = state["exp"]
            values = state["wavefront_values"] if mode == "Wavefront" else state["planning_values"]
            Plotting.plot_heat_map(
                exp.model,
                exp.observations,
                exp.actions,
                values,
                output_file=str(state["image"]),
                flip=flip,
                rotation=rotation,
                transition_weights=state["transition_weights"],
                edge_label_mode="int",
                vertex_label_mode="value",
                states=exp.decoded_states,
            )
            img_data = mpimg.imread(state["image"])
            if i < len(img_displays):
                img_displays[i].set_data(img_data)
            else:
                axes[i].axis("off")
                img_displays.append(axes[i].imshow(img_data, cmap="viridis"))
            axes[i].set_title(f"{exp.name}\n{mode}: t={state['step']}")

        def redraw_all():
            for i, state in enumerate(states):
                render(i, state)
            fig.canvas.draw_idle()

        def update_image(event):
            nonlocal mode
            if event.key == "q":
                plt.close(event.canvas.figure)
                return
            if event.key == "n":
                if mode == "Wavefront":
                    for state in states:
                        state["wavefront_values"], state["transition_weights"] = state["exp"].plan_method(
                            state["wavefront_values"],
                            state["transition_weights"],
                        )
                        state["step"] += 1
                else:
                    for state in states:
                        state["planning_values"] = Reasoning.propogate(
                            state["planning_values"],
                            state["transition_weights"],
                            state["planning_initial"],
                        )
                        state["step"] += 1
                        print(f"{state['exp'].name} chosen action", Reasoning.select_action(state["planning_values"], state["transition_weights"]), "\n")
                redraw_all()
            elif event.key == "m" and mode == "Wavefront":
                mode = "Planning"
                for state in states:
                    state["step"] = 0
                    state["planning_values"] = state["planning_initial"].copy()
                    print(f"{state['exp'].name} chosen action", Reasoning.select_action(state["planning_initial"], state["transition_weights"]), "\n")
                redraw_all()

        redraw_all()
        fig.suptitle("Experiment Comparison", fontsize=14)
        fig.text(
            0.5,
            0.02,
            "Controls: n - step | m - switch to planning | q - quit",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        fig.canvas.mpl_connect("key_press_event", update_image)
        if show:
            plt.show()
        return fig

    @staticmethod
    def graph_to_T(graph, normalize=True):
        """Convert a CS Academy graph text file into a single-action transition tensor.

        Parameters
        ----------
        graph : str
            File name under ``graphs/``. ``"graph1"`` and ``"graph1.txt"``
            both resolve to ``graphs/graph1.txt``.
            Paste graph text copied from https://csacademy.com/app/graph_editor/.
        normalize : bool
            If True, normalize each source node's outgoing edge weights so they
            sum to 1, making T a valid probability transition matrix. If False,
            edges with no explicit weight are set to 1 and explicit weights are
            kept as-is.

        Returns
        -------
        np.ndarray
            Transition tensor with shape ``(1, n_states, n_states)``.

        Notes
        -----
        Graph files use CS Academy's pasted text format: ``node`` for isolated
        nodes, ``src dst`` for directed edges, and ``src dst weight`` for
        weighted directed edges.
        """
        graph_path = Experiment._resolve_graph_path(graph)
        try:
            text = graph_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ValueError(f"Graph file not found: {graph_path}") from exc

        nodes = set()
        edges = []

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) not in (1, 2, 3):
                raise ValueError(
                    f"Malformed graph line {line_number}: expected 1, 2, or 3 numbers, got {len(parts)}."
                )

            if len(parts) == 1:
                node = Experiment._parse_non_negative_int(parts[0], line_number, "node")
                nodes.add(node)
                continue

            src = Experiment._parse_non_negative_int(parts[0], line_number, "source node")
            dst = Experiment._parse_non_negative_int(parts[1], line_number, "destination node")
            nodes.update((src, dst))
            weight = 1.0 if len(parts) == 2 else Experiment._parse_float(parts[2], line_number, "weight")
            edges.append({"src": src, "dst": dst, "weight": weight})

        if not nodes:
            raise ValueError(f"Graph file has no nodes or edges: {graph_path}")

        n_states = max(nodes) + 1
        T = np.zeros((1, n_states, n_states), dtype=float)
        for edge in edges:
            T[0, edge["src"], edge["dst"]] += edge["weight"]

        if normalize:
            return Experiment._normalize_T(T)
        return T

    @staticmethod
    def _resolve_graph_path(graph):
        if graph is None:
            raise ValueError("Graph file name is required.")

        graph_path = Path(graph)
        if graph_path.name != str(graph):
            raise ValueError("Graph must be a file name in graphs/, such as 'graph1' or 'graph1.txt'.")
        if graph_path.suffix == "":
            graph_path = graph_path.with_suffix(".txt")
        return Path("graphs") / graph_path

    @staticmethod
    def _parse_non_negative_int(value, line_number, label):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"Malformed graph line {line_number}: {label} must be an integer.") from exc
        if parsed < 0:
            raise ValueError(f"Malformed graph line {line_number}: {label} must be non-negative.")
        return parsed

    @staticmethod
    def _parse_float(value, line_number, label):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Malformed graph line {line_number}: {label} must be numeric.") from exc

    @staticmethod
    def _normalize_T(T):
        T = np.asarray(T, dtype=float).copy()
        norm = T.sum(axis=2, keepdims=True)
        norm[norm == 0] = 1
        return T / norm

    @staticmethod
    def _safe_filename(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "experiment"

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

    def _set_graph_model_context(self):
        n_states = self.T.shape[1]
        T = self.T
        model = types.SimpleNamespace(
            T=T,
            C=T,
            n_clones=np.ones(n_states, dtype=np.int64),
            state_observations=np.arange(n_states, dtype=np.int64) % DEFAULT_N_OBS,
        )
        model.decode = lambda obs, act: (None, np.arange(n_states, dtype=np.int64))
        self.model = model
        self.observations = np.arange(n_states, dtype=np.int64) % DEFAULT_N_OBS
        self.actions = np.zeros(n_states, dtype=np.int64)
        self.decoded_states = np.arange(n_states, dtype=np.int64)

    def _save_model(self):
        if self.model is None:
            return
        self.models_dir.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump((self.model, []), f, protocol=5)


def comparison(experiments, *args, **kwargs):
    return Experiment.comparison(experiments, *args, **kwargs)
