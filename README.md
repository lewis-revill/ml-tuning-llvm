# ml-tuning-llvm

## Overview

This repository contains scripts used to tune parameters internal to the LLVM register allocation algorithm in order to optimize for decreased code size. The machine learning techniques implemented by the scripts are: genetic algorithms (`ga_optimize.py`), simulated annealling (`sa_optimize.py`), and particle swarm optimization (`pso_optimize.py`). These tools currently can optimize for a RISC-V 32-bit processor.

## Usage

This project is set up such that it can optimize the collective size of C programs with the restriction that each program must consist of a set of C files within it's own subdirectory under `benchmarks/src`, plus any common header files required for all programs placed in the `benchmarks/support` directory. Recommended usage is to use the [Embench IOT](https://github.com/embench/embench-iot) benchmark suite as the source of these programs.

A [RISC-V toolchain](https://github.com/riscv/riscv-gnu-toolchain) must be built with 32-bit support, and a [modified LLVM compiler](https://github.com/lewis-revill/llvm-project/tree/ljr-regalloc-ml) (modifications on branch 'ljr-regalloc-ml') must be built, with `llc` available on the path, along with `riscv32-unknown-elf-clang` created as a symlink to `clang`.

Before running any optimization script, the `benchmarks/build_ir.py` script should be run.

Finally, either of the three optimization scripts can be used to tune the register allocation parameters.
