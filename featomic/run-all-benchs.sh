#!/usr/bin/env bash

set -eu

source ./virtualenv/bin/activate

mkdir -p memray && rm -f memray/*.bin

for file in "molecular_crystals" "silicon_bulk"; do
    echo  "================== $file =================="

    printf "QUIP\n"
    python bench-quip.py hypers.json $file.xyz no_grad
    memray stats memray/quip-$file.xyz-no_grad.bin | grep -A1 "Peak memory usage"

    printf "\nQUIP w/ gradients\n"
    python bench-quip.py hypers.json $file.xyz grad
    memray stats memray/quip-$file.xyz-grad.bin | grep -A1 "Peak memory usage"

    printf "\nlibrascal\n"
    python bench-librascal.py hypers.json $file.xyz no_grad
    memray stats memray/librascal-$file.xyz-no_grad.bin | grep -A1 "Peak memory usage"

    printf "\nlibrascal w/ gradients\n"
    python bench-librascal.py hypers.json $file.xyz grad
    memray stats memray/librascal-$file.xyz-grad.bin | grep -A1 "Peak memory usage"

    printf "\ndscribe\n"
    python bench-dscribe.py hypers.json $file.xyz no_grad
    memray stats memray/dscribe-$file.xyz-no_grad.bin | grep -A1 "Peak memory usage"

    printf "\ndscribe w/ gradients\n"
    python bench-dscribe.py hypers.json $file.xyz grad
    memray stats memray/dscribe-$file.xyz-grad.bin | grep -A1 "Peak memory usage"

    printf "\nspex CPU\n"
    python bench-spex.py hypers.json $file.xyz no_grad cpu
    memray stats memray/spex-$file.xyz-no_grad-cpu.bin | grep -A1 "Peak memory usage"

    printf "\nspex CUDA\n"
    python bench-spex.py hypers.json $file.xyz no_grad cuda
    memray stats memray/spex-$file.xyz-no_grad-cuda.bin | grep -A1 "Peak memory usage"

    printf "\nfeatomic\n"
    python bench-featomic.py hypers.json $file.xyz no_grad
    memray stats memray/featomic-$file.xyz-no_grad.bin | grep -A1 "Peak memory usage"

    printf "\nfeatomic w/ gradients\n"
    python bench-featomic.py hypers.json $file.xyz grad
    memray stats memray/featomic-$file.xyz-grad.bin | grep -A1 "Peak memory usage"
done
