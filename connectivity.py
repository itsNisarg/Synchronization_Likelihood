""" This script calculates the connectivity strength for a given band. The connectivity strength is saved in a .mat file."""
import os
import shutil
import mne
from synclike import Synchronization


band = input("Enter the band: ")
path = os.path.join("/..", "filtered", band + "_raw.fif.gz")
data = mne.io.read_raw_fif(path, preload=True)

# params = {"delta": {"m": 25, "lag": 41, "w1": 1968, "w2": 2967}, "theta": {"m": 7, "lag": 20, "w1": 240, "w2": 1239}, "alpha": {
#     "m": 6, "lag": 12, "w1": 120, "w2": 1119}, "beta": {"m": 8, "lag": 5, "w1": 70, "w2": 1069}, "gamma": {"m": 6, "lag": 3, "w1": 30, "w2": 1029}}

# sync = Synchronization(data.get_data(), *params[band])

# connectivity = sync.connectivity()
# print(connectivity.shape)

# shutil.move("connectivity.mat", band)
# shutil.move("sync", band)

print("DONE!")
