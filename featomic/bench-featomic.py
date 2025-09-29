import json
import sys
import time

import memray

import ase.io
from featomic import SoapPowerSpectrum


with open(sys.argv[1]) as fd:
    HYPERS = json.load(fd)


frames = ase.io.read(sys.argv[2], ":")
n_atoms = sum(len(f) for f in frames)

soap_hypers = {
    "cutoff": {
        "radius": HYPERS["cutoff"],
        "smoothing": {
            "type": "ShiftedCosine",
            "width": HYPERS["cutoff_width"],
        },
    },
    "density": {
        "type": "Gaussian",
        "width": HYPERS["atomic_width"],
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": HYPERS["max_angular"],
        "radial": {
            "type": "Gto",
            "max_radial": HYPERS["max_radial"],
        },
    },
}

calculator = SoapPowerSpectrum(**soap_hypers)

# warmup
for _ in range(3):
    _ = calculator.compute(frames)

if sys.argv[3] == "grad":
    gradients = ["positions"]
else:
    gradients = None


start = time.time()
for _ in range(HYPERS["n_iters"]):
    _ = calculator.compute(frames, gradients=gradients)
stop = time.time()
print(1e3 * (stop - start) / HYPERS["n_iters"] / n_atoms, "ms/atom")

memray_bin = f"memray/featomic-{sys.argv[2]}-{sys.argv[3]}.bin"
with memray.Tracker(memray_bin, native_traces=True):
    calculator = SoapPowerSpectrum(**soap_hypers)
    _ = calculator.compute(frames, gradients=gradients)
