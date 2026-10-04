"""Long-time (spectroscopy) cryoscope probe: vendor code only (qm/quam) — no qualibrate, no scqo.

Flux-line step response by qubit spectroscopy vs wait-time INTO a parked flux
pulse. Sequence per (detuning, wait): reset -> set the drive near the parked
qubit frequency (``df + IF + center_offset_hz``) -> play a long flux ``const``
pulse -> wait ``t`` into it -> play a long, weak SHAPED spectroscopy pulse
-> readout. Per wait-time the spectroscopy peak center gives the qubit frequency,
which maps to the delivered flux; the flux settling is fit downstream to a sum of
exponentials. This reaches the microsecond tails the Ramsey cryoscope cannot.

SPECTROSCOPY DRIVE: the tone is a RUN-SCOPED operation built by
:func:`install_drive_op` — the ``drive_shape`` envelope (square / cosine / gaussian)
at exactly ``operation_len_ns`` (on the 4 ns grid), at the amplitude that holds the
calibrated x180 ROTATION AREA (peak amplitude x length x envelope average/peak —
the envelope factor matters: see :func:`drive_amp_for_area`), so a longer pulse is
a proportionally weaker one and the tone stays a true pi pulse. It is played BARE:
no ``duration=`` stretch and no ``amplitude_scale``, so QUA's +/-2 ``amp()`` bound
never binds here. The line is then Fourier-limited at a constant set by the
envelope (FWHM x length = 0.799 square, 1.095 gaussian at sigma=T/4, 1.334 cosine)
— LENGTH sets the width, SHAPE sets how clean it is. A square tone's 11.6%
shoulders at +/- ~8.9 / length are RABI-NUTATION lobes (an AREA effect, not a sinc
sidelobe, so a flat top does not remove them) and they bias the estimator's
single-Lorentzian center by a few percent of the FWHM — the size of the settling
being tracked. The flux ``const`` is held for the wait PLUS this drive length plus
a fixed tail, so it always fully brackets the drive.

RUN-SCOPED OP LIFECYCLE: ``probe()`` files the built pulse on ``qubit.xy`` under
:data:`DRIVE_OP` and the shell's ``run()`` drops it in a ``finally``. The window is
exactly right: ``machine.generate_config()`` runs INSIDE ``_lib.acquire`` (after
``probe()`` has returned), while ``machine.save()`` — the only path that could
persist the op into ``state.json`` — runs much later, on accept/update. Same
recorded set -> revert discipline as
``scqo.experiments._drive_power.drive_power_boundary``. ``--preview`` does not go
through ``run()``, so the op is still on the tree when ``QMBackend.preview``
generates its config (the dumped script and the gateway waveform view are both
correct) and is left on the IN-MEMORY tree afterwards; a preview is its own process
and saves nothing, and :func:`install_drive_op` drops a stale copy first, so a
leftover can never be played.

PULSE CONTRACT: ``flux_amp_v`` is the parked flux amplitude in VOLTS measured
from the standing DC bias ``initialize_qpu`` applies — the z ``const`` rides on
that offset, so it is idle-relative. ``flux_point`` is passed EXPLICITLY because
the DAC emits ``idle + excursion`` and the number validated against the rail must
be the number that plays.

CENTER OFFSET: ``center_offset_hz`` is the arch-predicted parked detuning the
scqo experiment computes; the drive is shifted by it (NO LO shift — band-limited)
so the spectroscopy window stays centered on the peak. It is a plain frequency
shift added to ``update_frequency``.

Single qubit only: the sequence is built per qubit (one parked drive, one flux
line, one wait axis); run targets one at a time.

QM long-time (spectroscopy) cryoscope for scqo — supplies only ``probe()``.

Parameters, the spectroscopy-cryoscope estimator and the paired-fact writeback
are inherited from ``scqo.experiments.QubitSpectroscopyCryoscope``. scqo sweeps
the drive DETUNING (``detuning_hz``) x the (log-spaced) WAIT time
(``wait_time_ns``) into a parked flux pulse; the QM builder realizes it with a
held ``const`` flux, a wait, and a fixed ``x180`` spectroscopy pulse.

Unlike the Ramsey cryoscope, there is NO baking — a plain stretched ``const`` plus
the run-scoped drive op — so ``probe()`` returns ``(program, sweep_axes)`` and the
backend's shared fetch path runs it (no per-call baked config to thread through):
filing the op on the QUAM tree is enough, because the config is generated after
``probe()``.

Single target: the probe refuses more than one qubit (one parked drive + one flux
line + one wait axis); reset is resolved through ``_reset.check_reset_method`` so
``reset_method="active"`` is refused by name. The drive is centered on the
arch-predicted parked detuning ``resolved_center_offset_hz`` (no LO shift —
band-limited v1) and its envelope, length and amplitude come from ``drive_shape`` /
``drive_len_ns`` / ``drive_sigma_frac`` / ``drive_amp_factor``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from scqo_qm.experiments._flux_limits import check_flux_pulse_relative, idle_offset_v
from scqo_qm.experiments._lib import acquire as _acquire

#: extra flux-pulse cycles held past the spectroscopy pulse so the drive sits
#: fully inside the parked flux (4 ns cycles).
_FLUX_TAIL_CYCLES = 25

#: name of the RUN-SCOPED spectroscopy tone this probe files on the xy line. Built
#: per run at exactly ``drive_len_ns`` and dropped again in ``run()``'s ``finally``
#: — it is never meant to reach ``state.json`` (module docstring, RUN-SCOPED OP
#: LIFECYCLE).
DRIVE_OP = "spec_cryoscope_drive"


def validate_inputs(qubits, flux_amp_v: float, flux_point: str) -> float:
    """Pure pre-flight checks (no QUA); return the flux ``const`` reference amplitude.

    Refuses, by name and before any hardware time: more than one target (the
    sequence is built per qubit); a missing flux line; and a parked flux whose
    ``idle + flux_amp_v`` clips the port or needs an ``amplitude_scale`` QUA cannot
    express (:func:`check_flux_pulse_relative`).
    """
    if len(qubits) != 1:
        names = list(qubits.get_names())
        raise ValueError(
            f"qubit_spectroscopy_cryoscope builds its spectroscopy sequence per "
            f"qubit and supports one target at a time, got {len(qubits)}: {names}. "
            f"Run them one at a time.")
    qubit = qubits[0]
    z = getattr(qubit, "z", None)
    if z is None:
        raise ValueError(
            f"{qubit.name}: no flux line, but the spectroscopy cryoscope parks a z "
            f"pulse to detune the qubit — it cannot run on a qubit with no z channel.")
    return check_flux_pulse_relative(
        z, name=f"{qubit.name} spectroscopy cryoscope flux pulse",
        idle_v=idle_offset_v(z, flux_point), amps_v=[float(flux_amp_v)])


def _area_ratio(pulse, default: float) -> float:
    """``average/peak`` of the pulse's I-quadrature envelope — the rotation area
    a pulse delivers per unit (peak amplitude x length). 1.0 for a square pulse,
    0.5 for the lab's (Drag)Cosine x180. Sampled from the pulse's own
    ``waveform_function()`` so a different envelope stays correct; ``default``
    covers objects that cannot render samples (bare test stubs).
    """
    try:
        wf = np.asarray(pulse.waveform_function())
    except Exception:
        return default
    env = np.abs(np.real(wf)) if np.iscomplexobj(wf) else np.abs(wf)
    peak = float(np.max(env))
    if not np.isfinite(peak) or peak == 0.0:
        return default
    return float(np.mean(env) / peak)


def _mean_per_amplitude(pulse, default: float) -> float:
    """Mean of the pulse's I envelope per unit of its NOMINAL ``amplitude`` — the
    rotation area it delivers per (amplitude x ns).

    Deliberately NOT :func:`_area_ratio`'s average/PEAK. A ``subtracted`` gaussian
    renders a peak of only ``amplitude x (1 - exp(-length^2 / (8 sigma^2)))``
    (0.865 x at sigma = length/4), so dividing by the rendered peak and then
    STORING the result as ``amplitude`` leaves the tone 13.5% under a pi pulse —
    a silent under-rotation that just costs contrast. The nominal amplitude is what
    gets stored, so the nominal amplitude is what the ratio must be against.
    Sampled with the pulse at amplitude 1.0 (:func:`install_drive_op` builds it that
    way), so the mean IS the factor: 1.0 square, 0.5 raised cosine, ~0.462 gaussian
    at sigma = length/4.

    ``_area_ratio`` still measures the x180 REFERENCE, unchanged: its convention is
    what the calibrated pi amplitude has always been read against.
    """
    try:
        wf = np.asarray(pulse.waveform_function())
    except Exception:
        return default
    env = np.abs(np.real(wf)) if np.iscomplexobj(wf) else np.abs(wf)
    value = float(np.mean(env))
    return value if np.isfinite(value) and value > 0 else default


def drive_amp_for_area(x180_area: float, envelope_ratio: float,
                       length_ns: float, amp_factor: float = 1.0) -> float:
    """Peak amplitude holding the calibrated x180 ROTATION AREA over ``length_ns``.

    ``x180_area`` is the x180's own ``average/peak x amplitude x length``;
    ``envelope_ratio`` is the tone's mean envelope per unit NOMINAL amplitude
    (:func:`_mean_per_amplitude`: 1.0 square, 0.5 raised cosine, ~0.462 gaussian at
    sigma = length/4), so the returned number can be stored as ``amplitude``
    directly. The envelope
    factor is load-bearing: the x180 is a cosine-envelope DRAG pulse that delivers
    only HALF its peak x length, so transplanting the bare product onto a square
    tone doubles the area — an exact 2*pi pulse, whose spectroscopy response is
    DARK on resonance with two bright shoulders at +/- sqrt(1.25)/length (seen on
    hardware as a split line). With the ratio in, the tone is a true pi pulse: full
    contrast on resonance and a Fourier-limited line, and holding the area means a
    longer pulse is a proportionally weaker one (no power broadening).

    Pure arithmetic (no QUA, no QUAM), so it is unit-tested directly.
    """
    return float(amp_factor) * float(x180_area) / (
        float(envelope_ratio) * float(length_ns))


def make_drive_pulse(shape: str, length_ns: int, sigma_frac: float = 0.25):
    """The unscaled QUAM pulse realizing ``shape`` over ``length_ns`` — amplitude
    1.0, since :func:`install_drive_op` sets the real one once the attached
    envelope can be sampled.

    ``square`` renders a CONSTANT waveform (no instrument waveform memory);
    ``cosine`` and ``gaussian`` render ARBITRARY waveforms, one sample per ns.
    ``cosine`` is ``FlatTopCosinePulse`` with a zero flat top, i.e. rise and fall
    each half the length — exactly a raised-cosine (Hann) envelope, the shape
    family of the lab's DragCosine x180 with the DRAG quadrature left off (and
    without its baked Stark detuning, which would move the line).
    """
    from quam.components.pulses import (
        FlatTopCosinePulse,
        GaussianPulse,
        SquarePulse,
    )

    length = int(length_ns)
    if shape == "square":
        return SquarePulse(length=length, amplitude=1.0, axis_angle=0.0)
    if shape == "cosine":
        return FlatTopCosinePulse(length=length, amplitude=1.0, flat_length=0,
                                  axis_angle=0.0)
    if shape == "gaussian":
        return GaussianPulse(length=length, amplitude=1.0,
                             sigma=float(sigma_frac) * length, axis_angle=0.0,
                             subtracted=True)
    raise ValueError(
        f"drive_shape={shape!r} is not realized on the QM backend; use 'square', "
        f"'cosine' or 'gaussian'.")


def check_drive_amp(amplitude: float, xy, *, name: str) -> None:
    """Refuse BY NAME a tone louder than the loudest already stored on this xy line.

    The empirical headroom statement: the line is already configured to emit its
    loudest stored operation, so a peak at or under that one is proven safe. A
    waveform at or above the port's full scale CLIPS on hardware while the QM
    simulator shows nothing — the drive twin of the rail check in ``_flux_limits``.
    It does not live in ``_amp_limits`` on purpose: that module answers "can
    ``amp()`` represent this multiplier?", while this is a port-emission question.
    """
    peaks = [abs(float(op.amplitude)) for op in xy.operations.values()
             if isinstance(getattr(op, "amplitude", None), (int, float))]
    if not peaks:
        return
    ceiling = max(peaks)
    if abs(float(amplitude)) > ceiling:
        raise ValueError(
            f"{name}: the pi-area drive amplitude {abs(float(amplitude)):.4f} is "
            f"louder than {ceiling:.4f}, the loudest tone already stored on that xy "
            f"line, so it risks clipping the port (silent on the simulator). "
            f"Lengthen drive_len_ns or lower drive_amp_factor.")


def install_drive_op(qubit, *, shape: str, length_ns: int, sigma_frac: float = 0.25,
                     amp_factor: float = 1.0) -> str:
    """File the run-scoped spectroscopy tone on ``qubit.xy``; return its op name.

    Drops any stale copy first (idempotent — a ``--preview`` leaves one behind),
    attaches the pulse BEFORE sampling it so ``Pulse._get_sampling_rate()`` resolves
    through the channel instead of warning and defaulting to 1 GHz, then sets the
    amplitude that holds the x180 rotation area (:func:`drive_amp_for_area`) and
    checks the headroom (:func:`check_drive_amp`). Paired with :func:`drop_drive_op`
    — see the module docstring's RUN-SCOPED OP LIFECYCLE.
    """
    xy = qubit.xy
    drop_drive_op(qubit)
    pulse = make_drive_pulse(shape, length_ns, sigma_frac)
    xy.operations[DRIVE_OP] = pulse

    x180 = xy.operations["x180"]
    x180_area = _area_ratio(x180, default=0.5) * x180.amplitude * x180.length
    # sampled at amplitude 1.0, so the mean IS the area per (amplitude x ns)
    ratio = _mean_per_amplitude(pulse, default=1.0 if shape == "square" else 0.5)
    amplitude = drive_amp_for_area(x180_area, ratio, length_ns, amp_factor)
    check_drive_amp(amplitude, xy, name=f"{qubit.name} spectroscopy cryoscope drive")
    pulse.amplitude = amplitude
    return DRIVE_OP


def drop_drive_op(qubit) -> None:
    """Remove the run-scoped tone from ``qubit.xy`` — idempotent, never raises."""
    ops = getattr(getattr(qubit, "xy", None), "operations", None)
    if ops is not None:
        ops.pop(DRIVE_OP, None)


def build_program(
    machine,
    qubits,
    *,
    dfs,
    wait_cycles,
    flux_amp_v: float,
    center_offset_hz: float,
    num_shots: int,
    reset_type: str = "thermal",
    operation: str = DRIVE_OP,
    operation_len_ns: int = 400,
    use_state_discrimination: bool = False,
    simulate: bool = False,
    flux_point: str = "joint",
    log: Optional[Callable] = None,
):
    """Build the long-time cryoscope QUA program. Returns ``(program, sweep_axes)``.

    ``dfs`` is the drive-detuning sweep (Hz), swept OUTER; ``wait_cycles`` the
    wait-into-the-flux sweep in clock cycles (4 ns), swept INNER (typically
    log-spaced). ``flux_amp_v`` is the idle-relative parked amplitude and
    ``center_offset_hz`` the arch-predicted parked detuning the drive is shifted
    by. ``operation`` is the RUN-SCOPED spectroscopy tone :func:`install_drive_op`
    already filed on the xy line at exactly ``operation_len_ns`` and at the
    x180-area amplitude, so it is played BARE — no ``duration=``, no
    ``amplitude_scale``; ``operation_len_ns`` is passed only so the flux ``const``
    can bracket it. No baking — a plain stretched ``const`` plus that pulse — so
    the shared ``_lib.acquire`` fetches it (the adapter returns ``(prog, axes)``).
    """
    amp_ref = validate_inputs(qubits, flux_amp_v, flux_point)
    qubit = qubits[0]
    dfs = np.round(np.asarray(dfs)).astype(int)
    wait_cycles = np.unique(np.asarray(wait_cycles).astype(int))
    const_scale = float(flux_amp_v) / amp_ref
    offset_hz = int(round(float(center_offset_hz)))
    # Long, weak spectroscopy tone, already built at this exact length and at the
    # x180-area amplitude by install_drive_op; the cycle count is what the flux
    # `const` below has to bracket.
    op_len_cycles = int(operation_len_ns) // 4
    if operation not in qubit.xy.operations:
        raise ValueError(
            f"{qubit.name}: xy operation {operation!r} is not on the line — the "
            f"spectroscopy cryoscope builds it per run, so probe() must call "
            f"install_drive_op() before build_program().")

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning_hz": xr.DataArray(
            dfs.astype(float), attrs={"long_name": "drive detuning", "units": "Hz"}),
        "wait_time_ns": xr.DataArray(
            (4 * wait_cycles).astype(float),
            attrs={"long_name": "wait into the flux pulse", "units": "ns"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        df = declare(int)
        t = declare(int)
        if use_state_discrimination:
            state = declare(int)
            state_st = declare_stream()

        machine.initialize_qpu(target=qubit, flux_point=flux_point)
        align()

        with for_(n, 0, n < num_shots, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                with for_each_(t, wait_cycles):
                    qubit.reset(reset_type, simulate, log_callable=log)
                    # park the drive at the arch-predicted detuned frequency + the swept df
                    qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + offset_hz)
                    align()
                    # hold the flux on while the drive waits t into it, then probes
                    qubit.z.play("const", amplitude_scale=const_scale,
                                 duration=t + op_len_cycles + _FLUX_TAIL_CYCLES)
                    qubit.xy.wait(t)
                    qubit.xy.play(operation)
                    align()
                    if use_state_discrimination:
                        qubit.readout_state(state)
                        save(state, state_st)
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                        save(I[0], I_st[0])
                        save(Q[0], Q_st[0])

        with stream_processing():
            n_st.save("n")
            if use_state_discrimination:
                state_st.buffer(len(wait_cycles)).buffer(len(dfs)).average().save("state1")
            else:
                I_st[0].buffer(len(wait_cycles)).buffer(len(dfs)).average().save("I1")
                Q_st[0].buffer(len(wait_cycles)).buffer(len(dfs)).average().save("Q1")

    return prog, sweep_axes


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log: Optional[Callable] = None,
) -> xr.Dataset:
    """Connect to the QOP, execute the program and fetch the raw xr.Dataset."""
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log)


from typing import Any, ClassVar

import numpy as np
from scqo import register
from scqo.experiments import QubitSpectroscopyCryoscope


@register
class QMQubitSpectroscopyCryoscope(QubitSpectroscopyCryoscope):
    """Build the long-time cryoscope spectroscopy sweep on the QM OPX."""

    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_spectroscopy_cryoscope --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "the spectroscopy pulse is an operation built on the drive line for this run "
        "and removed afterwards; it is refused when louder than the loudest pulse "
        "already stored on that line",
        "the area of the x180 is read from its sampled envelope, so the amplitude "
        "follows the shape the x180 really has",
    )

    def run(self):
        """``super().run()`` in a ``try``, dropping the run-scoped drive op in the
        ``finally`` — the recorded set -> revert discipline of
        ``scqo.experiments._drive_power.drive_power_boundary``.

        The op has to survive ``probe()`` (the config is generated afterwards,
        inside ``_lib.acquire``) but must be gone before any ``machine.save()``, so
        the whole acquire-and-estimate bracket is the right teardown scope.
        """
        try:
            return super().run()
        finally:
            self._drop_drive_op()

    def _drop_drive_op(self) -> None:
        """Best-effort teardown: a backend that never reached the machine (an
        offline preflight failure) must not turn into a second exception here."""
        machine = getattr(self.backend, "machine", None)
        qubits = getattr(machine, "qubits", None) or {}
        for target in self.params.targets:
            qubit = qubits.get(str(target))
            if qubit is not None:
                drop_drive_op(qubit)

    def probe(self) -> Any:
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.quam_fields import GOVERNED_FLUX_POINT

        from ._reset import check_reset_method

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)
        target = str(self.params.targets[0])
        # ns (on the 4 ns grid) -> clock cycles for the QUA wait loop
        wait_ns = self.sweep_axes["wait_time_ns"]
        wait_cycles = np.maximum(4, np.round(wait_ns / 4)).astype(int)

        # the run-scoped spectroscopy tone: built at exactly drive_len_ns and at
        # the pi-area amplitude, dropped again in run()'s finally.
        drive_len_ns = int(self.params.drive_len_ns)
        operation = install_drive_op(
            qubits[0],
            shape=str(self.params.drive_shape),
            length_ns=drive_len_ns,
            sigma_frac=float(self.params.drive_sigma_frac),
            amp_factor=float(self.params.drive_amp_factor),
        )

        return build_program(
            machine,
            qubits,
            dfs=self.sweep_axes["detuning_hz"],
            wait_cycles=wait_cycles,
            flux_amp_v=float(self.params.flux_pulse_amp_v),
            center_offset_hz=self.resolved_center_offset_hz(target),
            operation=operation,
            operation_len_ns=drive_len_ns,
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            flux_point=GOVERNED_FLUX_POINT,
        )
