# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
# MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
One thread's memory for every engine, kept from pair to pair: an `Aligner`'s, a batch worker's and a C aligner's.

Each engine owns the memory it reuses (`gap_affine.WavefrontSpace`, `edit_distance.EditSpace`, `ksw_ties.TieBuffers`);
this module, above them all, holds them together, so no engine has to know the others'.
"""

from .edit_distance import EditSpace
from .errors import AlignmentError
from .gap_affine import WavefrontSpace
from .penalties import Penalties
from .ksw_ties import TieBuffers
from .modes import Costs


struct SearchSpace(Movable):
    """A thread's engines' memory: the wavefront's searches (`wave`), the edit distance's (`edit`, whose tally
    holds every engine's probes in a build with probes) and `Ties.KSW2`'s writing again (`ties`)."""

    var wave: WavefrontSpace
    var edit: EditSpace
    var ties: TieBuffers

    def __init__(out self):
        """Nothing built yet: the first pair builds what it needs."""
        self.wave = WavefrontSpace()
        self.edit = EditSpace()
        self.ties = TieBuffers()

    def penalties_for(mut self, costs: Costs) raises AlignmentError -> Penalties:
        """The wavefront's penalties for `costs`, worked out once for a batch's costs (see
        `WavefrontSpace.penalties_for`)."""
        return self.wave.penalties_for(costs)

    def tally_wavefronts(mut self, started: Int):
        """The wavefronts' probes since the last call into the tally, with the time since `started` (see
        `WavefrontSpace.tally_into`)."""
        self.wave.tally_into(self.edit.tally, started)
