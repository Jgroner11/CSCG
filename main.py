import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from CSCG_helpers import Plotting, Reasoning
from experiment import Experiment
from rooms import DONUT

# for experiment_name in Experiment.list_experiments():
#     print(f"- {experiment_name}")


exp = Experiment(
    name="asymmetric_loop",
    graph="loop",
    graph_normalize=True,
    graph_third_column="weight",
    plan_method=Reasoning.STP,
    starts=[0],
    targets=[3],
    wavefront_steps=4,
    planning_steps=4,
)

exp.run()

output_file = os.path.join("figures", f"{exp.name}-graph.png")
Plotting.plot_graph(
    exp.model,
    exp.observations,
    exp.actions,
    output_file=output_file,
    states=exp.decoded_states,
    flip = True,
    rotation=.9
)
image = mpimg.imread(output_file)
fig, ax = plt.subplots()
ax.axis("off")
ax.imshow(image)
plt.show()

exp.visualize(mode="combined")



# Interactive visualization requires a CSCG model context, so this graph-only
# example just saves its transition/planning artifacts under experiments/.
