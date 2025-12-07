#!/usr/bin/env python3
import torch
from typing import Dict, List, Optional

import metatomic.torch as mta
import metatensor.torch as mts


def f_coord(y: torch.Tensor) -> torch.Tensor:
    """
    This function computes a switching function that we use
    to evaluate the coordination number.
    """
    cy = torch.zeros_like(y)
    # we use torch.where to be compatible with autodiff
    cy = torch.where(y <= 0, torch.tensor(1.0, dtype=torch.float32), cy)
    cy = torch.where(y >= 1, torch.tensor(0.0, dtype=torch.float32), cy)
    mask = (y > 0) & (y < 1)
    cy = torch.where(mask, ((y - 1) ** 2) * (1 + 2 * y), cy)
    return cy


class CoordinationHistogram(torch.nn.Module):
    def __init__(self, cutoff, cn_list):
        """
        ``cutoff`` is the neighbor list cutoff, used for metatomic.

        ``cn_list`` is the list of bins in the histogram.
        """
        super().__init__()

        # Physical parameters for Argon in Angstroms
        sigma_lj = 3.4
        self.r0 = 1.5 * sigma_lj
        self.r1 = 1.3 * sigma_lj

        self.cn_list = torch.tensor(cn_list, dtype=torch.int32)

        # Ensure the neighbor list is large enough for the r0 cutoff
        self._nl_options = mta.NeighborListOptions(
            cutoff=max(cutoff, self.r0 + 0.1), full_list=True, strict=True
        )

    def requested_neighbor_lists(self) -> List[mta.NeighborListOptions]:
        return [self._nl_options]

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[mts.Labels],
    ) -> Dict[str, mts.TensorMap]:

        # Handle boilerplate for empty systems, etc.
        if "features" not in outputs:
            return {}
        if outputs["features"].per_atom:
            raise ValueError("per-atoms features are not supported in this model")
        if selected_atoms is not None:
            raise ValueError("selected_atoms is not supported in this model")
        if len(systems[0].positions) == 0:
            keys = mts.Labels("_", torch.tensor([[0]]))
            block = mts.TensorBlock(
                torch.zeros((0, len(self.cn_list)), dtype=torch.float64),
                samples=mts.Labels("structure", torch.zeros((0, 1), dtype=torch.int32)),
                components=[],
                properties=mts.Labels("cn", self.cn_list.reshape(-1, 1)),
            )
            return {"features": mts.TensorMap(keys, [block])}

        values = []
        system_index = torch.arange(len(systems), dtype=torch.int32).reshape((-1, 1))

        for system in systems:
            neighbors = system.get_neighbor_list(self._nl_options)
            distances = torch.linalg.vector_norm(neighbors.values.reshape(-1, 3), dim=1)

            # Per-atom coordination number (c_j)
            z = f_coord((distances - self.r1) / (self.r0 - self.r1))
            coords = torch.zeros(len(system), dtype=z.dtype, device=z.device)
            coords.index_add_(dim=0, index=neighbors.samples.column("first_atom"), source=z)

            num_atoms = len(system)
            cn_list_tensor = self.cn_list.to(device=coords.device, dtype=coords.dtype)
            diffs = coords - cn_list_tensor.reshape(-1, 1)

            # ##########################################################################
            # # Integrated kernel (Quadratic B-spline) for a normalized CV
            # ##########################################################################
            abs_diffs = torch.abs(diffs)
            kernel_vals = torch.zeros_like(diffs)
            # Region 1: |u| <= 0.5
            mask1 = abs_diffs <= 0.5
            kernel_vals = torch.where(mask1, 0.75 - abs_diffs**2, kernel_vals)
            # Region 2: 0.5 < |u| <= 1.5
            mask2 = (abs_diffs > 0.5) & (abs_diffs <= 1.5)
            kernel_vals = torch.where(mask2, 0.5 * (1.5 - abs_diffs)**2, kernel_vals)
            # ##########################################################################

            if num_atoms > 0:
                cn_histo = torch.sum(kernel_vals, dim=1)
            else:
                cn_histo = torch.zeros_like(cn_list_tensor)
            values.append(cn_histo)

        # Assemble and return the TensorMap
        keys = mts.Labels("_", torch.tensor([[0]]))
        block = mts.TensorBlock(
            values=torch.stack(values, dim=0),
            samples=mts.Labels("structure", system_index),
            components=[],
            properties=mts.Labels("cn", self.cn_list.reshape(-1, 1)),
        )
        return {"features": mts.TensorMap(keys, [block])}

# generates a coordination histogram model
cutoff = 5.5
module = CoordinationHistogram(cutoff, cn_list=[6, 8])

# metatdata about the model itself
metadata = mta.ModelMetadata(
    name="Coordination histogram",
    description="Computes smooth histogram of coordination numbers",
)

# metatdata about what the model can do
capabilities = mta.ModelCapabilities(
    length_unit="Angstrom",
    outputs={"features": mta.ModelOutput(per_atom=False)},
    atomic_types=[18],  # Ar
    interaction_range=cutoff,
    supported_devices=["cpu"],
    dtype="float64",
)

model_ch = mta.AtomisticModel(
    module=module.eval(),
    metadata=metadata,
    capabilities=capabilities,
)

model_ch.save("gen/histo-cv.pt", collect_extensions="./extensions/")
