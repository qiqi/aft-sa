#!/bin/bash
# Double-precision AVL rebuild (the near-field CDind is a small difference
# of large terms and NaNs/inflates in float32 at this AR -- Appendix D).
# Produces tools/avl/Avl/bin/avl. Limits raised: NVMAX 6000->8000, NSMAX
# 500->700 in src/AVL.INC.
set -e
cd "$(dirname "$0")/../../tools/avl/Avl"
(cd plotlib && cp config.make.gfortran config.make && make gfortranDP)
(cd eispack && gfortran -O2 -fdefault-real-8 -c eispack.f && ar r eispack.a eispack.o)
cd bin && make -f Makefile.gfortran FC=gfortran \
  FFLAGS="-O2 -fdefault-real-8 -std=legacy -mcmodel=medium" \
  PLTOBJ=../plotlib/libPlt_gDP.a EIGOBJ=../eispack/eispack.a avl
