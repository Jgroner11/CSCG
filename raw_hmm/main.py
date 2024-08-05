from matplotlib import pyplot as plt
import numpy as np
from HMM import MM, HMM

mm = MM(5, ('u', 3, False))
# mm = MM(7, ('r', 3, True))
mm.plot_path()


hmm = HMM(7, 3, A=("u", 3, True), B = ("u", 3, True))
hmm.save_model('uniform_hmm')

hmm = HMM.load_model('uniform_hmm')
print('state frequencies:', [float(round(x, 2)) for x in hmm.sf])
print('observation frequencies:', [float(round(x, 2)) for x in hmm.of])
hmm.plot_path()


A = np.array(
    [
        [0, 1, 0],
        [.5, 0, .5],
        [0, 1, 0]
    ]
)

B = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
)
hmm = HMM(3, 3, A, B)

Os = [
        [0, 0],
        [0, 1],
        [0, 1, 0],
        [0, 1, 2],
        [2, 0]
    ]
for O in Os:
    print(O, 'using alpha:', hmm.compute_obs_sequence_probability(O, using='alpha'), 'using beta:', hmm.compute_obs_sequence_probability(O, using='beta'))
hmm.plot_path()


