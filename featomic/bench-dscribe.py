import json
import sys
import time
import os

import memray

import ase.io
from dscribe.descriptors import SOAP


with open(sys.argv[1]) as fd:
    HYPERS = json.load(fd)

frames = ase.io.read(sys.argv[2], ":")

all_types = set()
for frame in frames:
    all_types.update(frame.numbers)
n_species = len(all_types)

n_atoms = sum(len(f) for f in frames)

calculator = SOAP(
    species=list(all_types),
    periodic=True,
    r_cut=HYPERS["cutoff"],
    n_max=HYPERS["max_radial"] + 1,
    l_max=HYPERS["max_angular"],
    sigma=HYPERS["atomic_width"],
    rbf="gto",
)

do_grad = sys.argv[3] == "grad"

# warmup
for _ in range(3):
    if do_grad:
        _ = calculator.derivatives(frames, n_jobs=os.cpu_count())
    else:
        _ = calculator.create(frames, n_jobs=os.cpu_count())


start = time.time()
for _ in range(HYPERS["n_iters"]):
    if do_grad:
        _ = calculator.derivatives(frames, n_jobs=os.cpu_count())
    else:
        _ = calculator.create(frames, n_jobs=os.cpu_count())
stop = time.time()
print(1e3 * (stop - start) / HYPERS["n_iters"] / n_atoms, "ms/atom")

memray_bin = f"memray/dscribe-{sys.argv[2]}-{sys.argv[3]}.bin"
with memray.Tracker(memray_bin, native_traces=True):
    calculator = SOAP(
        species=list(all_types),
        periodic=True,
        r_cut=HYPERS["cutoff"],
        n_max=HYPERS["max_radial"] + 1,
        l_max=HYPERS["max_angular"],
        sigma=HYPERS["atomic_width"],
        rbf="gto",
    )
    if do_grad:
        _ = calculator.derivatives(frames, n_jobs=os.cpu_count())
    else:
        _ = calculator.create(frames, n_jobs=os.cpu_count())
