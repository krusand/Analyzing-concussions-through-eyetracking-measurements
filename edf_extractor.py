
import pyedfread as pedf

samples, events, messages = pedf.read_edf("Data/anti-saccade/87/anti-saccade_97_0.EDF")

print(samples)

