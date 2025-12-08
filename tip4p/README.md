# Tip4p Model Simulations

This directory contains LAMMPS input scripts to run molecular dynamics simulations of
water using the Q-TIP4P model. Two different simulation setups are provided: one using
the native LAMMPS implementation of the water model (`lmp.in`) and another using the
metatomic TIP4P implementation (`mts.in`).

In order to run these simulations, you will need to compile the metatomic LAMMPS version
as described in the [metatomic
documentation](https://docs.metatensor.org/metatomic/latest/engines/lammps.html#engine-lammps).

For the metatomic implementation, you will need the **model file** (`qtip4pf-mta.pt`). To create this file follow the instructions in the atomistic cookbook chapter
on [Atomistic Water Model for Molecular
Dynamics](https://atomistic-cookbook.org/examples/water-model/water-model.html#using-the-q-tip4p-f-model).
