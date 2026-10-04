"""What only this backend consumes, on top of an experiment's neutral requirements.

The core declares what every backend needs (``scqo/requirements.py``,
``scqo/experiments/_requires.py``). A knob only this driver's probes read is
declared here and added by the driver's subclass:

    requires = (*QubitEcho.requires, X90_AMPLITUDE)

``scqo run <name> --help`` then lists it with the rest.
"""

from __future__ import annotations

from scqo.requirements import Requirement

#: QM plays a pi/2 pulse as an operation of its own (x90, y90, ...) with its own
#: stored amplitude - it is not half of the pi pulse's
X90_AMPLITUDE = Requirement(
    "pi_amp_x90", "the amplitude of the pi/2 operations (x90, y90) this probe plays")
