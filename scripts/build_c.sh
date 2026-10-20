#!/usr/bin/env bash
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Builds karan's C API (see c/karan.h) into build/c: libkaran for the platform's oldest CPU, or
# `target-cpu`, beside the Mojo runtime libraries it loads and the header.
#
#     pixi run build-c [target-cpu]
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
# The file is not c/karan.mojo, which would be a module of the package's own name importing itself.
BUNDLE_NAME=karan bash "$root/scripts/bundle.sh" --library "$root/c/karan_c.mojo" "$root/build/c" "$@"
cp "$root/c/karan.h" "$root/build/c/"
