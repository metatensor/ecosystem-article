import ase.io
import numpy as np

minimal = ase.io.read("data/lj-oct.xyz")
lammps_initial = minimal.copy()
lammps_initial.cell = [30, 30, 30]
lammps_initial.center()
lammps_initial.pbc = np.array([False, False, False])
ase.io.write("gen/minimal.data", lammps_initial, format="lammps-data")
