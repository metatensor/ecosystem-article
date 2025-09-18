import json
import resource
import sys
import time

import ase.io
import spex.metatensor
import vesin.torch
import torch


with open(sys.argv[1]) as fd:
    HYPERS = json.load(fd)

frames = ase.io.read(sys.argv[2], ":")
n_atoms = sum(len(f) for f in frames)


all_types = set()
for frame in frames:
    all_types.update(frame.numbers)

calculator = spex.metatensor.SoapPowerSpectrum(
    cutoff=HYPERS["cutoff"],
    max_angular=HYPERS["max_angular"],
    radial={"LaplacianEigenstates": {"max_radial": HYPERS["max_radial"]}},
    angular="SphericalHarmonics",
    species={"Orthogonal": {"species": list(all_types)}},
    cutoff_function={"ShiftedCosine": {"width": HYPERS["cutoff_width"]}},
)

nl = vesin.torch.NeighborList(cutoff=HYPERS["cutoff"], full_list=True)


def compute(calculator, frames, do_grad, device):
    calculator.to(device)

    all_rij = []
    all_i = []
    all_j = []
    all_species = []
    structure = []
    center = []

    all_positions = []

    start = 0
    for i_frame, frame in enumerate(frames):
        positions = torch.tensor(frame.positions, requires_grad=do_grad)
        rij, i, j = nl.compute(
            positions,
            torch.tensor(frame.cell[:]),
            periodic=True,
            quantities="Dij",
        )

        all_positions.append(positions)

        all_rij.append(rij)
        all_i.append(i + start)
        all_j.append(j + start)
        all_species.append(torch.tensor(frame.numbers))

        start += len(frame)

        structure.append(torch.full((len(frame),), i_frame))
        center.append(torch.arange(len(frame)))

    soap = calculator(
        torch.cat(all_rij).to(device),
        torch.cat(all_i).to(device),
        torch.cat(all_j).to(device),
        torch.cat(all_species).to(device),
        torch.cat(structure).to(device),
        torch.cat(center).to(device),
    )

    if do_grad:
        raise Exception("gradients of SOAP are very slow with autodiff")
        # for block in soap:
        #     print("here")
        #     grad_outputs = []
        #     for row in range(block.values.shape[0]):
        #         go = torch.zeros_like(block.values)
        #         go[row] = 1.0
        #         grad_outputs.append(go.reshape(-1, *go.shape))

        #     gradients = torch.autograd.grad(
        #         outputs=block.values,
        #         inputs=all_positions,
        #         grad_outputs=torch.vstack(grad_outputs),
        #         is_grads_batched=True,
        #     )

    return soap


do_grad = sys.argv[3] == "grad"
device = sys.argv[4]

# warmup
for _ in range(3):
    _ = compute(calculator, frames, do_grad, device)

start = time.time()
for _ in range(HYPERS["n_iters"]):
    _ = compute(calculator, frames, do_grad, device)
stop = time.time()
print(1e3 * (stop - start) / HYPERS["n_iters"] / n_atoms, "ms/atom")


if sys.platform.startswith("linux"):
    max_mem_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
else:
    max_mem_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

print(max_mem_mib / n_atoms, "MiB/atom on CPU")

if device == "cuda":
    max_memory_cuda_mib = torch.cuda.memory.max_memory_allocated(device) / 1024 / 1024
    print(max_memory_cuda_mib / n_atoms, "MiB/atom on GPU")
