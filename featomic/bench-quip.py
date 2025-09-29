import json
import sys
import time

import memray

import ase.io
import quippy


with open(sys.argv[1]) as fd:
    HYPERS = json.load(fd)

frames = ase.io.read(sys.argv[2], ":")
n_atoms = sum(len(f) for f in frames)

all_types = set()
for frame in frames:
    all_types.update(frame.numbers)
n_species = len(all_types)

quip_soap_hypers = [
    "soap",
    f"n_max={HYPERS['max_radial'] + 1}",
    f"l_max={HYPERS['max_angular']}",
    f"cutoff={HYPERS['cutoff']}",
    f"cutoff_transition_width={HYPERS['cutoff_width']}",
    f"atom_sigma={HYPERS['atomic_width']}",
    f"n_species={n_species}",
    "species_Z={" + " ".join(map(str, all_types)) + "}",
]

soap_calculator = quippy.descriptors.Descriptor(" ".join(quip_soap_hypers))

do_grad = sys.argv[3] == "grad"

# warmup
for _ in range(3):
    _ = soap_calculator.calc(frames, grad=do_grad)

start = time.time()
for _ in range(HYPERS["n_iters"]):
    _ = soap_calculator.calc(frames, grad=do_grad)
stop = time.time()
print(1e3 * (stop - start) / HYPERS["n_iters"] / n_atoms, "ms/atom")


memray_bin = f"memray/quip-{sys.argv[2]}-{sys.argv[3]}.bin"
with memray.Tracker(memray_bin, native_traces=True):
    soap_calculator = quippy.descriptors.Descriptor(" ".join(quip_soap_hypers))
    _ = soap_calculator.calc(frames, grad=do_grad)
