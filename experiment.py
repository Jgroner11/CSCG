class Experiment:
    def __init__(self):
        pass

    def run(self):
        if not self.exists(self.name):
            pass
            # save to experiments folder under folder name

    def visualize(self):
        pass

    def add_note(self, txt):
        pass

    @staticmethod
    def list_experiments():
        pass

    @staticmethod
    def exists(name):
        pass

def comparison(experiments):
    for exp in experiments:
        exp.view()
