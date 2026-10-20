#!/usr/bin/env bash
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Builds the command-line aligner (see cli/karan_cli.mojo) into build/cli as `karan`, for
# the platform's oldest CPU or `target-cpu`, beside the runtime libraries it loads, so the folder runs
# without pixi or Mojo.
#
#     pixi run build-cli [target-cpu]
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
bash "$root/scripts/bundle.sh" "$root/cli/karan_cli.mojo" "$root/build/cli" "$@"
mv "$root/build/cli/karan_cli" "$root/build/cli/karan"
echo "$root/build/cli/karan"
