#!/bin/bash

VERSION=3.13.2
PREFIX=$PWD/Ipopt-releases-${VERSION}

wget -c https://github.com/coin-or/Ipopt/archive/releases/${VERSION}.tar.gz -O - | tar -xz
cd Ipopt-releases-${VERSION}
./configure --prefix=${PREFIX} --with-asl
make install -j4

echo "Done. Set environment: export PATH=${PREFIX}/bin:\$PATH"

exit 0
