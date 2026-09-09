# AlphaPROBE Python package

This directory contains the installable AlphaPROBE mining package and its
research baselines. For architecture, configuration, operations, and known
limitations, see the [repository README](../../README.md).

The existing project metadata and `pdm.lock` target an independent Python 3.11
and CUDA 12.1 environment. The pinned torch, torchvision, and torchaudio wheels
are CPython 3.11/CUDA 12.1 builds and must not be mixed into the platform's
Python 3.12 environment. Changing CPU/CUDA or Python targets requires a formal
lock refresh.
