# -*- mode:snakemake; -*-

import ase.io
from ase.mep import NEB


# Depends on the endpoint alignment (which depends on minimization)
rule generate_idpp_images:
    input:
        reactant=f"{config['paths']['endpoints']}/reactant.con",
        product=f"{config['paths']['endpoints']}/product.con",
    output:
        # Declare all N+2 output files that will be generated.
        expand(
            f"{config['paths']['idpp']}/path/{{num:02d}}.con",
            num=range(config["common"]["number_of_intermediate_imgs"] + 2),
        ),
    params:
        niimgs=config["common"]["number_of_intermediate_imgs"],
    run:
        react = ase.io.read(input.reactant)
        prod = ase.io.read(input.product)

        images = [react]
        images += [react.copy() for i in range(params.niimgs)]
        images += [prod]

        neb = NEB(images)
        neb.interpolate("idpp")

        # Zip function ensures we match the correct image to the correct output path
        for outfile, img in zip(output, images):
            ase.io.write(outfile, img)


# Summary file for EON to use
rule collect_paths:
    input:
        # Depends on the output of the rule above.
        expand(
            f"{config['paths']['idpp']}/path/{{num:02d}}.con",
            num=range(config["common"]["number_of_intermediate_imgs"] + 2),
        ),
    output:
        f"{config['paths']['idpp']}/idppPath.dat",
    shell:
        # List the absolute paths of the inputs.
        "realpath {input} > {output}"
