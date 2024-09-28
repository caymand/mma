#!/bin/sh
FRAGS_M=$1
FRAGS_N=$2
FRAGS_K=$3
WARP_TILES_M=$4
WARP_TILES_N=$5
WARP_TILES_K=$6
BLOCK_TILES_M=$7
BLOCK_TILES_N=$8


command="nvcc -O3 -std=c++17 -arch=sm_80 -Xptxas=-v --expt-relaxed-constexpr -lcublas -I../../cutlass/include/ main.cu -o main"


if [ $1 = "--" ]
then
  shift 1
  $command "$@"
else
  shift 8
  set -x
  $command \
    -DFRAGS_M=$FRAGS_M \
    -DFRAGS_N=$FRAGS_N \
    -DFRAGS_K=$FRAGS_K \
    -DWARP_TILES_M=$WARP_TILES_M \
    -DWARP_TILES_N=$WARP_TILES_N \
    -DWARP_TILES_K=$WARP_TILES_K \
    -DBLOCK_TILES_M=$BLOCK_TILES_M \
    -DBLOCK_TILES_N=$BLOCK_TILES_N \
    "$@"
fi

