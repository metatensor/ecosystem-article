# -*- mode:snakemake; -*-


rule do_minimization:
    input:
        config="resources/config_minim.ini",
        endpoint=f"{config['paths']['endpoints']}/{{endpoint}}_pre_aligned.con",
        model=expand(
            f"{config['paths']['models']}/pet-mad-{{version}}.pt",
            version=config["pet_mad"]["version"],
        ),
    output:
        endpoint=f"{config['paths']['endpoints']}/{{endpoint}}_minimized.con",
    shadow:
        "minimal"
    shell:
        """
        cp {input.config} config.ini
        cp {input.endpoint} pos.con
        cp {input.model} .
        eonclient
        cp min.con {output.endpoint}
        """


rule do_neb:
    input:
        config="resources/config_neb.ini",
        idpp_path=f"{config['paths']['idpp']}/idppPath.dat",
        reactant=f"{config['paths']['endpoints']}/reactant.con",
        product=f"{config['paths']['endpoints']}/product.con",
        path_images=expand(
            f"{config['paths']['idpp']}/path/{{num:02d}}.con",
            num=range(config["common"]["number_of_intermediate_imgs"] + 2),
        ),
        model=expand(
            f"{config['paths']['models']}/pet-mad-{{version}}.pt",
            version=config["pet_mad"]["version"],
        ),
    output:
        results_dat=f"{config['paths']['neb']}/results.dat",
        neb_con=f"{config['paths']['neb']}/neb.con",
        neb_dat=f"{config['paths']['neb']}/neb.dat",
        all_neb_dat=expand(
            f"{config['paths']['neb']}/neb_{{num:03d}}.dat",
            num=range(config["eon"]["num_steps"]),
        ),
        all_neb_con=expand(
            f"{config['paths']['neb']}/neb_path_{{num:03d}}.con",
            num=range(config["eon"]["num_steps"]),
        ),
    params:
        opath=config["paths"]["neb"],
    shadow:
        "minimal"
    shell:
        """
        cp {input.model} {params.opath}/
        cp {input.config} {params.opath}/config.ini
        cp {input.idpp_path} {params.opath}/
        cp {input.reactant} {params.opath}/reactant.con
        cp {input.product} {params.opath}/product.con
        cd {params.opath}
        eonclient 2>&1 || true
        """
