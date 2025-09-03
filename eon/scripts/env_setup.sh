#!/usr/bin/env bash

export SETUPTOOLS_SCM_PRETEND_VERSION

SETUPTOOLS_SCM_PRETEND_VERSION="v0.0.dev0+metatensor_ecosystem_$(git rev-parse HEAD)"

export GITROOT

GITROOT=$(git rev-parse --show-toplevel)
