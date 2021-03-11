#!/usr/bin/env python3

import logging as log
import os
import subprocess


llc = 'llc'
llcflags = [
    '-x=mir', '-filetype=obj',
    '-enable-machine-outliner',
    '-mattr=+save-restore',
    '-regalloc=greedy', '-start-before=greedy',
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


def evaluate_benchmarks(additional_llcargs):
  builddir = os.path.join(rootdir, 'bd')

  # Find all the machine IR benchmarks present.
  benchmarks = os.listdir(builddir)

  log.debug('Found benchmarks: {benchmarks}'.format(benchmarks=benchmarks))

  # Compile each machine IR benchmark and evaluate the code size.
  product = 1.0
  count = 0
  for benchmark in benchmarks:
    bench_builddir = os.path.join(builddir, benchmark)

    # Find the machine IR source files.
    mirfiles = [f for f in os.listdir(bench_builddir) \
                    if os.path.splitext(f)[1].lower() == '.mir']
    log.debug('Found machine IR files {mirfiles} for {benchmark}'
                 .format(mirfiles=mirfiles, benchmark=benchmark))

    for mirfile in mirfiles:
      mirpath = os.path.join(bench_builddir, mirfile)

      # Construct LLC compile command from args and input/output files.
      llcarglist = [llc]
      llcarglist.extend(llcflags)
      llcarglist.extend(additional_llcargs)
      llcarglist.extend([mirpath])
      llcarglist.extend(['-o', '-'])

      log.debug('Compiling machine IR {mirfile} for {benchmark}'
                   .format(mirfile=mirfile, benchmark=benchmark))
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
              'Warning: Compilation of {mirpath} failed'.format(mirpath=mirpath)
          )
          succeeded = False
      except subprocess.TimeoutExpired:
        log.warning('Warning: Compilation of {mirpath} timed out'
                        .format(mirpath=mirpath))
        succeeded = False

      if not succeeded:
        log.debug(res.stdout.decode('utf-8'))
        log.debug(res.stderr.decode('utf-8'))

      # Estimate the size by the number of bytes in the output object file.
      # Subtract the number of bytes in a completely empty object file to
      # improve accuracy of this estimate.
      size = len(res.stdout) - 584
      product *= size
      count += 1

  geomean = pow(product, 1 / count)
  return geomean


def main():
  # Setup logging.
  log.basicConfig(level=log.INFO)

  total_size = evaluate_benchmarks(additional_llcargs=[])

  log.info(total_size)


if __name__ == '__main__':
  main()
