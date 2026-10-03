import mne
import matplotlib.pyplot as plt

raw = mne.io.read_raw_edf(
    "chb03_01.edf",
    preload=False
)

# Show the region around the seizure
raw.plot(
    start=330,
    duration=120,
    n_channels=10,
    scalings="auto",
    block=True
)

plt.show()