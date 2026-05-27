from CSCG_helpers import Reasoning
from experiment import Experiment, GRANULAR_ROOM


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

print("chosen actions:", exp1.chosen_actions)
print("action plan:", exp1.action_plan)
