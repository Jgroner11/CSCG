from experiment import Experiment
from CSCG_helpers import Reasoning

# for experiment_name in Experiment.list_experiments():
#     print(f"- {experiment_name}")

# Experiment.comparison(["loop", "dloop"])

if __name__ == "__main__":
    exp = Experiment.get("granular")
    exp.visualize(
        plan_method = Reasoning.STP4
    )
