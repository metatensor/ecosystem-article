# -*- mode:snakemake; -*-

import ase.io
import ira_mod
import numpy as np
import copy


rule prepare_endpoints_pre:
    input:
        reactant="resources/reactant.con",
        product="resources/product.con",
    output:
        reactant=f"{config['paths']['endpoints']}/reactant_pre_aligned.con",
        product=f"{config['paths']['endpoints']}/product_pre_aligned.con",
    run:
        # Load reactant and product
        try:
            reactant_atm = ase.io.read(input.reactant)
            product_atm = ase.io.read(input.product)
        except FileNotFoundError as e:
            raise Exception(
                f"Error: {e}. Make sure reactant.con and product.con are present."
            )

            # Center them
        for atm in [reactant_atm, product_atm]:
            atm.set_cell([25, 25, 25])
            atm.center()

        print("Read and centered reactant.con and product.con.")

        # Prepare data for the IRA library
        nat1 = len(reactant_atm)
        typ1 = reactant_atm.get_atomic_numbers()
        coords1 = reactant_atm.get_positions()

        nat2 = len(product_atm)
        typ2 = product_atm.get_atomic_numbers()
        coords2 = product_atm.get_positions()

        if nat1 != nat2:
            print(
                f"Warning: Reactant ({nat1} atoms) and Product ({nat2} atoms) "
                "have different atom counts. IRA will match the largest common subset."
            )

            # Initialize and run IRA shape-matching
        ira = ira_mod.IRA()
        kmax_factor = 1.8  # Default value

        print("Running ira.match to find rotation, translation, AND permutation...")
        # r = rotation, t = translation, p = permutation, hd = Hausdorff distance
        r, t, p, hd = ira.match(nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor)

        print(f"Matching complete. Hausdorff Distance (hd) = {hd:.6f} Å")

        # Create the fully aligned and permuted product structure
        coords2_aligned = (coords2 @ r.T) + t
        coords2_aligned_permuted = coords2_aligned[p]

        # Save the new aligned-and-permuted structure
        aligned_product_atm = copy.deepcopy(reactant_atm)
        aligned_product_atm.positions = coords2_aligned_permuted

        # Save files
        ase.io.write(output.product, aligned_product_atm)
        ase.io.write(output.reactant, reactant_atm)

        diff_sq = (coords1 - coords2_aligned_permuted) ** 2
        rmsd = np.sqrt(np.mean(np.sum(diff_sq, axis=1)))
        print(f"Pre-minimization RMSD: {rmsd:.6f} Å")


rule prepare_endpoints_post:
    input:
        reactant=f"{config['paths']['endpoints']}/reactant_minimized.con",
        product=f"{config['paths']['endpoints']}/product_minimized.con",
    output:
        reactant=f"{config['paths']['endpoints']}/reactant.con",
        product=f"{config['paths']['endpoints']}/product.con",
    run:
        # Load minimized reactant and product
        try:
            reactant_atm = ase.io.read(input.reactant)
            product_atm = ase.io.read(input.product)
        except FileNotFoundError as e:
            raise Exception(f"Error: {e}. Make sure minimized endpoints are present.")

        print("Read minimized reactant and product.")

        # Prepare data for the IRA library
        nat1 = len(reactant_atm)
        typ1 = reactant_atm.get_atomic_numbers()
        coords1 = reactant_atm.get_positions()

        nat2 = len(product_atm)
        typ2 = product_atm.get_atomic_numbers()
        coords2 = product_atm.get_positions()

        if nat1 != nat2:
            print(
                f"Warning: Reactant ({nat1} atoms) and Product ({nat2} atoms) "
                "have different atom counts. IRA will match the largest common subset."
            )

            # Initialize and run IRA shape-matching
        ira = ira_mod.IRA()
        kmax_factor = 1.8

        print("Running ira.match to find rotation, translation, AND permutation...")
        # r = rotation, t = translation, p = permutation, hd = Hausdorff distance
        r, t, p, hd = ira.match(nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor)

        print(f"Matching complete. Hausdorff Distance (hd) = {hd:.6f} Å")

        # Create the fully aligned and permuted product structure
        # Apply rotation (r) and translation (t) to the original product coordinates
        # This aligns the product's orientation to the reactant's
        coords2_aligned = (coords2 @ r.T) + t

        # Apply the permutation (p)
        # This re-orders the aligned product atoms to match the reactant's atom order
        # p[i] = j means reactant atom 'i' matches product atom 'j'
        # So, the new coordinate array's i-th element should be coords2_aligned[j]
        coords2_aligned_permuted = coords2_aligned[p]

        # Save the new aligned-and-permuted structure
        aligned_product_atm = copy.deepcopy(reactant_atm)
        aligned_product_atm.positions = coords2_aligned_permuted

        # Save files
        ase.io.write(output.product, aligned_product_atm)
        ase.io.write(output.reactant, reactant_atm)

        diff_sq = (coords1 - coords2_aligned_permuted) ** 2
        rmsd = np.sqrt(np.mean(np.sum(diff_sq, axis=1)))
        print(f"Post-minimization RMSD: {rmsd:.6f} Å")
