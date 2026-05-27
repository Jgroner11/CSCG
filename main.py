import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from CSCG_helpers import Plotting, Reasoning
from experiment import Experiment
from rooms import SIMPLE_GRANULAR_ROOM

for experiment_name in Experiment.list_experiments():
    print(f"- {experiment_name}")


# exp = Experiment(
#     name="navigation-simple_granular_room",
#     room=SIMPLE_GRANULAR_ROOM,
#     plan_method=Reasoning.STP,
#     wavefront_steps=10,
#     planning_steps=10,
# )

# exp.run()


# exp = Experiment.get("navigation-simple_granular_room")
# output_file = os.path.join("figures", f"{exp.name}-graph.png")
# Plotting.plot_graph(
#     exp.model,
#     exp.observations,
#     exp.actions,
#     output_file=output_file,
#     states=exp.decoded_states,
#     flip = True,
#     rotation=.9
# )
# image = mpimg.imread(output_file)
# fig, ax = plt.subplots()
# ax.axis("off")
# ax.imshow(image)
# plt.show()


Experiment.get("navigation-simple_granular_room").visualize(
    mode="combined",
    starts=[10],
    targets=[28],
)
