#!/usr/bin/env python3

import logging as log
import os
import subprocess


cc = 'riscv32-unknown-elf-clang'
cflags = [
    '-S', '-emit-llvm',
    '-Oz',
    '-mllvm', '-enable-machine-outliner',
    '-msave-restore',
    '-march=rv32imc', '-mabi=ilp32',
    '-fdata-sections', '-ffunction-sections',
    '-mllvm', '-regalloc=greedy',
    '-ffixed-x3',
    '-ffixed-x4',
    '-ffixed-x5',
    '-ffixed-x6',
    '-ffixed-x7',
    '-ffixed-x9',
    '-ffixed-x18',
    '-ffixed-x19',
    '-ffixed-x20',
    '-ffixed-x21',
    '-ffixed-x22',
    '-ffixed-x23',
    '-ffixed-x24',
    '-ffixed-x25',
    '-ffixed-x26',
    '-ffixed-x27',
    '-ffixed-x28',
    '-ffixed-x29',
    '-ffixed-x30',
    '-ffixed-x31',
]
llc = 'llc'
llcflags = [
    '-enable-machine-outliner',
    '-mattr=+save-restore',
    '-regalloc=greedy', '-stop-before=greedy',
    '-mattr=+reserve-x3',
    '-mattr=+reserve-x4',
    '-mattr=+reserve-x5',
    '-mattr=+reserve-x6',
    '-mattr=+reserve-x7',
    '-mattr=+reserve-x9',
    '-mattr=+reserve-x18',
    '-mattr=+reserve-x19',
    '-mattr=+reserve-x20',
    '-mattr=+reserve-x21',
    '-mattr=+reserve-x22',
    '-mattr=+reserve-x23',
    '-mattr=+reserve-x24',
    '-mattr=+reserve-x25',
    '-mattr=+reserve-x26',
    '-mattr=+reserve-x27',
    '-mattr=+reserve-x28',
    '-mattr=+reserve-x29',
    '-mattr=+reserve-x30',
    '-mattr=+reserve-x31',
]

rootdir = os.path.abspath(os.path.dirname(__file__))


def main():
  # Setup logging.
  log.basicConfig(level=log.INFO)

  srcdir = os.path.join(rootdir, 'src')
  builddir = os.path.join(rootdir, 'bd')

  # Find all the benchmarks present.
  benchmarks = os.listdir(srcdir)
  log.debug('Found benchmarks: {benchmarks}'.format(benchmarks=benchmarks))

  # Create directories for output build files.
  for benchmark in benchmarks:
    bench_builddir = os.path.join(builddir, benchmark)
    try:
      os.makedirs(bench_builddir)
    except PermissionError:
      log.warning(
          'Warning: Unable to create build directory for {bench_builddir}'
              .format(bench_builddir=bench_builddir)
      )
      return False
    except FileExistsError:
      pass

  # Compile each benchmark to LLVM IR, then to pre-regalloc machine IR.
  for benchmark in benchmarks:
    bench_srcdir = os.path.join(srcdir, benchmark)
    bench_builddir = os.path.join(builddir, benchmark)

    # Find the source files.
    srcfiles = [f for f in os.listdir(bench_srcdir) \
                    if os.path.splitext(f)[1].lower() == '.c']
    log.debug('Found source files {srcfiles} for {benchmark}'
                 .format(srcfiles=srcfiles, benchmark=benchmark))

    # Compile each source file to an output LLVM IR file, then to pre-regalloc machine IR.
    for srcfile in srcfiles:
      srcpath = os.path.join(bench_srcdir, srcfile)
      irfile = os.path.splitext(srcfile)[0] + '.ll'
      irpath = os.path.join(bench_builddir, irfile)

      # Construct CC compile command from args and input/output files.
      ccarglist = [cc]
      ccarglist.extend(cflags)
      ccarglist.extend(['-I{rootdir}/support'.format(rootdir=rootdir)])
      ccarglist.extend(['-DCPU_MHZ=1', '-DWARMUP_HEAT=1'])
      ccarglist.extend([srcpath])
      ccarglist.extend(['-o', irpath])

      log.debug('Compiling source {srcfile} to IR for {benchmark}'
                   .format(srcfile=srcfile, benchmark=benchmark))
      log.debug('Compile arguments:\n{ccarglist}'.format(ccarglist=ccarglist))

      # Run compilation.
      res = None
      succeeded = True
      try:
        res = subprocess.run(
            ccarglist,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=500,
        )
        if res.returncode != 0:
          log.warning(
              'Warning: Compilation of {srcpath} failed'.format(srcpath=srcpath)
          )
          succeeded = False
      except subprocess.TimeoutExpired:
        log.warning('Warning: Compilation of {srcpath} timed out'
                        .format(srcpath=srcpath))
        succeeded = False

      if not succeeded:
        log.debug(res.stdout.decode('utf-8'))
        log.debug(res.stderr.decode('utf-8'))

      mirfile = os.path.splitext(srcfile)[0] + '.mir'
      mirpath = os.path.join(bench_builddir, mirfile)

      # Construct LLC compile command from args and input/output files.
      llcarglist = [llc]
      llcarglist.extend(llcflags)
      llcarglist.extend([irpath])
      llcarglist.extend(['-o', mirpath])

      log.debug('Compiling IR {irfile} to machine IR for {benchmark}'
                   .format(irfile=irfile, benchmark=benchmark))
      log.debug('LLC arguments:\n{llcarglist}'.format(llcarglist=llcarglist))

      # Run LLC.
      res = None
      succeeded = True
      try:
        res = subprocess.run(
            llcarglist,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=500,
        )
        if res.returncode != 0:
          log.warning(
              'Warning: Compilation of {irpath} failed'.format(irpath=irpath)
          )
          succeeded = False
      except subprocess.TimeoutExpired:
        log.warning(
            'Warning: Compilation of {irpath} timed out'.format(irpath=irpath)
        )
        succeeded = False

      if not succeeded:
        log.debug(res.stdout.decode('utf-8'))
        log.debug(res.stderr.decode('utf-8'))


if __name__ == '__main__':
  main()
