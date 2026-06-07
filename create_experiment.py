import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from CSCG_helpers import Plotting, Reasoning
from experiment import Experiment
from rooms import DONUT, GRANULAR_ROOM, SIMPLE_GRANULAR_ROOM


if __name__ == "__main__":
    exp = Experiment(
        name="simple_room",
        room=SIMPLE_GRANULAR_ROOM,
        plan_method=Reasoning.STP1,
        starts=[73],
        targets=[28],
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