from CSCG_helpers import Reasoning
from experiment import Experiment
from visualize_stp import GRANULAR_ROOM


exp1 = Experiment(
    name="navigation-granular_room",
    room=GRANULAR_ROOM,
    plan_method=Reasoning.STP,
    starts=[52],
    targets=[42],
    wavefront_steps=10,
    planning_steps=10,
)

exp1.run()

exp1.visualize(mode="combined")
