import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from CSCG_helpers import Plotting, Reasoning
from experiment import Experiment
from rooms import DONUT

# for experiment_name in Experiment.list_experiments():
#     print(f"- {experiment_name}")


# exp = Experiment(
#     name="navigation-donut",
#     room=DONUT,
#     plan_method=Reasoning.STP,
#     seq_length=5000,
#     n_clones=25,
# )

# exp.run()


# exp = Experiment.get("navigation-donut")
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


Experiment.get("navigation-donut").visualize(
    mode="combined",
    starts=[23],
    targets=[70],
)
