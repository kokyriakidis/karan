#!/usr/bin/env bash
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Builds the Python package (see python/karan) into build/python: the package's Python, its
# extension `_karan` for the platform's oldest CPU or `target-cpu`, and the Mojo runtime libraries it
# loads, so `PYTHONPATH=build/python python3 -c "import karan"` works without pixi or Mojo.
#
#     pixi run build-python [target-cpu]
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
out=$root/build/python/karan
mkdir -p "$out"
bash "$root/scripts/bundle.sh" --library "$root/python/_karan.mojo" "$out" "$@"
# Python imports an extension by its module's name, the bundle names a library `lib<name>`.
for library in "$out"/lib_karan.so "$out"/lib_karan.dylib; do
    [ -e "$library" ] && mv "$library" "$out/_karan.so"
done
cp "$root/python/karan/"*.py "$out/"
echo "$out: import karan with $(dirname "$out") on PYTHONPATH"
