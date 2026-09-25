"""Adaptive Bayesian T1 probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Berritta et al., arXiv:2506.09576, in the u = 1/k parametrization. Per block:
``num_probes`` adaptive single shots — prepare |e>, wait tau = c * T1_est from
the CURRENT posterior, measure, conditionally flip back — each followed by the
on-FPGA method-of-moments update of the posterior state (T1, u). Optionally
each adaptive probe is chased by one NON-adaptive shot on a linear wait grid
(the classical-decay cross-check).

Numeric-range rationale (QUA ``fixed`` is signed 4.28, range [-8, 8)):

* Times are carried in MILLISECONDS internally so T1 up to ~8 ms fits.
* The posterior SHAPE k never fits: the paper's k reaches tens, so the state
  is tracked as u = 1/k (always small) and k is NEVER materialized. The update
  is reciprocal-free — u' = (1+u) * ratio_{k+1}/ratio_k - 1 — and the only
  k-exponent quantity, r^k, is computed as exp(-c * phi(z)) with z = c*u and
  phi(z) = ln(1+z)/z (Taylor series below z = 0.2, Math.ln/div above), an O(1)
  number for any k.
* ms -> clock cycles MUST use ``Cast.mul_int_by_fixed(250_000, tau_ms)``: the
  fixed-arithmetic product ``tau_ms * 250000`` exceeds 8 and WRAPS (modulo 16)
  to a garbage wait — the classic silent fixed-point failure.
* The SPAM-aware likelihood reads alpha = P(read 0 | prep 1) and
  beta = P(read 1 | prep 0) — the readout channel's measured assignment
  fidelities, alpha = 1 - fidelity_e and beta = 1 - fidelity_g (the rows of a
  confusion matrix sum to 1). The shell resolves them and refuses a qubit whose
  fidelities are unmeasured BEFORE any QUA is built; the builder takes numbers.

Heterogeneous streams (per-block scalars, (block, probe) arrays, evolution
vectors, a timestamp stream) rule out ``_lib.acquire`` / XarrayDataFetcher,
so the probe ships its own ``acquire()`` (the ``qubit_tomography`` pattern).

QM adaptive Bayesian T1 for scqo - supplies only ``probe()``.

Parameters, the credible-interval analysis and reporting are inherited from
``scqo.experiments.QubitT1Bayesian``. The probe streams heterogeneous shapes,
so ``probe()`` returns the 3-tuple ``(program, sweep_axes, probe_module)`` and
the backend uses the probe module's OWN ``acquire()``.

Two vendor-side prerequisites, each refused BY NAME before any QUA is built:

* a calibrated readout threshold (the sequence discriminates every shot);
* the readout channel's ``fidelity_g`` / ``fidelity_e`` monitors, which
  ``single_shot_readout`` stores on every successful run: the SPAM-aware
  likelihood needs alpha = P(read 0 | prep 1) = 1 - fidelity_e and
  beta = P(read 1 | prep 0) = 1 - fidelity_g. They are scqo state, measured on
  THIS setup, so the run that produced them is provenance a QUAM-side matrix
  never had. Both are staleness-gated the same way every stored readout
  reference is: they describe the readout condition they were measured at.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import xarray as xr
from qm.qua import *

US_TO_MS = 1e-3
MS_TO_CLK_INT = 250_000  # ms -> 4 ns clock cycles, int form for Cast.mul_int_by_fixed

#: the shortest adaptive wait, ms (1 us) — matches scqo's TAU_MIN_S.
TAU_MIN_MS = 1e-3


def build_program(
    machine,
    qubits,
    *,
    num_blocks: int,
    num_probes: int,
    c_adaptive: float,
    k0: float,
    t1_prior_s: Dict[str, float],
    spam_pairs: Dict[str, tuple],
    t1_min_s: float,
    t1_max_s: float,
    k_min: float,
    k_max: float,
    interleaved: bool,
    lin_wait_cycles,
    active_reset_per_probe: bool,
    reset_type: str,
    reset_max_attempts: int = 15,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the adaptive Bayesian T1 QUA program. Returns (program, sweep_axes).

    ``t1_prior_s`` maps qubit name -> prior mean T1 in seconds (resolved by the
    shell through scqo's one prior door); ``lin_wait_cycles`` is the validation
    grid in 4 ns cycles (length ``num_probes``).
    """
    num_qubits = len(qubits)
    names = qubits.get_names()

    # bound k by clamping u in [1/k_max, 1/k_min]; additionally 1 + c*u_max
    # must stay inside the fixed range because it feeds Math.ln / Math.inv
    u_max_safe = (7.5 - 1.0) / c_adaptive
    k_min_val = max(float(k_min), 1.0 / u_max_safe)
    u_min = 1.0 / float(k_max)
    u_max = 1.0 / k_min_val
    u0 = float(np.clip(1.0 / float(k0), u_min, u_max))
    t1_min_ms = float(t1_min_s) * 1e3
    t1_max_ms = float(t1_max_s) * 1e3
    tau_max_ms = t1_max_ms
    prior_ms = [float(np.clip(t1_prior_s[name] * 1e3, t1_min_ms, t1_max_ms))
                for name in names]
    lin_wait_cycles = np.maximum(4, np.asarray(lin_wait_cycles, dtype=int))

    sweep_axes = {
        "qubit": xr.DataArray(names),
        "block_idx": xr.DataArray(np.arange(num_blocks)),
    }

    with program() as prog:
        _, _, _, _, n, n_st = machine.declare_qua_variables()
        rep = declare(int)
        probe = declare(int)

        state = [declare(int) for _ in range(num_qubits)]
        state_lin = [declare(int) for _ in range(num_qubits)]

        # posterior state per qubit: (t1_est [ms], u = 1/k). k is NEVER stored.
        u_inv_k = [declare(fixed) for _ in range(num_qubits)]
        t1_est = [declare(fixed) for _ in range(num_qubits)]
        alpha = [declare(fixed) for _ in range(num_qubits)]
        beta = [declare(fixed) for _ in range(num_qubits)]

        tau_ms = [declare(fixed) for _ in range(num_qubits)]
        tau_clocks = [declare(int) for _ in range(num_qubits)]

        z = declare(fixed)       # z = tau/theta = c*u (O(1), always in range)
        z2 = declare(fixed)
        phi = declare(fixed)     # phi(z) = ln(1+z)/z
        lexp = declare(fixed)    # L = k*ln(r) = -c*phi(z)
        r = declare(fixed)
        r_pow_k = declare(fixed)
        r_pow_k1 = declare(fixed)
        r_pow_k2 = declare(fixed)
        spam = declare(fixed)
        num_k = declare(fixed)
        den_k = declare(fixed)
        num_k1 = declare(fixed)
        den_k1 = declare(fixed)
        ratio_k = declare(fixed)
        ratio_k1 = declare(fixed)
        ratio_ratio = declare(fixed)

        lin_tau = declare(int, value=lin_wait_cycles.tolist())

        t_est_st = [declare_stream() for _ in range(num_qubits)]
        u_final_st = [declare_stream() for _ in range(num_qubits)]
        u_evol_st = [declare_stream() for _ in range(num_qubits)]
        t_evol_st = [declare_stream() for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        tau_st = [declare_stream() for _ in range(num_qubits)]
        state_lin_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for i, qubit in multiplexed_qubits.items():
                # SPAM likelihood terms, resolved by the shell from the readout
                # channel's measured assignment fidelities (build-time numbers).
                spam_alpha, spam_beta = spam_pairs[qubit.name]
                assign(alpha[i], float(spam_alpha))  # P(read 0 | prep 1)
                assign(beta[i], float(spam_beta))    # P(read 1 | prep 0)

            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(rep, 0, rep < num_blocks, rep + 1):
                save(rep, n_st)

                for i, qubit in multiplexed_qubits.items():
                    assign(u_inv_k[i], u0)
                    assign(t1_est[i], prior_ms[i])
                    assign(state[i], 0)

                for i, qubit in multiplexed_qubits.items():
                    qubit.reset(reset_type, simulate, log_callable=log,
                                max_attempts=reset_max_attempts)
                align()

                with for_(probe, 0, probe < num_probes, probe + 1):
                    # tau_i = c * T1_est from the current posterior
                    for i, qubit in multiplexed_qubits.items():
                        with if_(rep == num_blocks - 1):
                            # posterior entering this probe of the LAST block
                            save(u_inv_k[i], u_evol_st[i])
                            save(t1_est[i], t_evol_st[i])
                        assign(tau_ms[i], c_adaptive * t1_est[i])
                        with if_(tau_ms[i] < TAU_MIN_MS):
                            assign(tau_ms[i], TAU_MIN_MS)
                        with if_(tau_ms[i] > tau_max_ms):
                            assign(tau_ms[i], tau_max_ms)
                        # ms -> cycles via int*fixed: `tau_ms * 250000` in
                        # fixed arithmetic wraps (modulo 16) to a garbage wait
                        assign(tau_clocks[i], Cast.mul_int_by_fixed(MS_TO_CLK_INT, tau_ms[i]))
                        with if_(tau_clocks[i] < 4):
                            assign(tau_clocks[i], 4)

                    # prepare |e>, wait, measure, flip back on outcome |1>
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x180")
                        qubit.align()
                        qubit.resonator.wait(tau_clocks[i])
                        qubit.readout_state(state[i])
                        qubit.align()
                        qubit.xy.play("x180", condition=Cast.to_bool(state[i]))
                        save(state[i], state_st[i])
                        save(tau_ms[i], tau_st[i])
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        # method-of-moments update (Eqs. 5-6) in u = 1/k form:
                        #   ratio_k  = [num @ k] / [den @ k]
                        #   T1'      = T1 / ratio_k
                        #   u'       = (1+u) * ratio_{k+1}/ratio_k - 1
                        # r^k = exp(-c * phi(z)), z = c*u, phi(z) = ln(1+z)/z
                        assign(spam, 1.0 - alpha[i] - beta[i])
                        assign(z, c_adaptive * u_inv_k[i])
                        assign(r, Math.inv(1.0 + z))  # 1+z >= 1, safe for inv
                        with if_(z < 0.2):
                            # phi(z) Taylor: err < 1e-4 for z < 0.2, no division
                            assign(z2, z * z)
                            assign(phi, 1.0 - 0.5 * z + z2 * (1.0 / 3.0) - (z2 * z) * (1.0 / 4.0))
                        with else_():
                            # z >= 0.2 > 1/8, so Math.div is safe here
                            assign(phi, Math.div(Math.ln(1.0 + z), z))
                        assign(lexp, -c_adaptive * phi)
                        assign(r_pow_k, Math.exp(lexp))
                        assign(r_pow_k1, r_pow_k * r)
                        assign(r_pow_k2, r_pow_k1 * r)

                        with if_(state[i] == 0):
                            assign(num_k, 1.0 - beta[i] - spam * r_pow_k1)
                            assign(den_k, 1.0 - beta[i] - spam * r_pow_k)
                            assign(num_k1, 1.0 - beta[i] - spam * r_pow_k2)
                            assign(den_k1, 1.0 - beta[i] - spam * r_pow_k1)
                        with else_():
                            assign(num_k, beta[i] + spam * r_pow_k1)
                            assign(den_k, beta[i] + spam * r_pow_k)
                            assign(num_k1, beta[i] + spam * r_pow_k2)
                            assign(den_k1, beta[i] + spam * r_pow_k1)

                        assign(ratio_k, Math.div(num_k, den_k))
                        assign(ratio_k1, Math.div(num_k1, den_k1))

                        assign(t1_est[i], Math.div(t1_est[i], ratio_k))
                        with if_(t1_est[i] < t1_min_ms):
                            assign(t1_est[i], t1_min_ms)
                        with if_(t1_est[i] > t1_max_ms):
                            assign(t1_est[i], t1_max_ms)

                        assign(ratio_ratio, Math.div(ratio_k1, ratio_k))
                        assign(u_inv_k[i], (1.0 + u_inv_k[i]) * ratio_ratio - 1.0)
                        with if_(u_inv_k[i] < u_min):
                            assign(u_inv_k[i], u_min)
                        with if_(u_inv_k[i] > u_max):
                            assign(u_inv_k[i], u_max)

                    if interleaved:
                        # the adaptive probe's conditional x180 already left
                        # |g>; re-flipping on `state` here would bias the whole
                        # non-adaptive curve. Optionally run a full reset.
                        if active_reset_per_probe:
                            for i, qubit in multiplexed_qubits.items():
                                qubit.reset(reset_type, simulate, log_callable=log,
                                            max_attempts=reset_max_attempts)
                            align()

                        for i, qubit in multiplexed_qubits.items():
                            assign(tau_clocks[i], lin_tau[probe])
                            qubit.xy.play("x180")
                        align()

                        for i, qubit in multiplexed_qubits.items():
                            qubit.resonator.wait(tau_clocks[i])
                        align()

                        for i, qubit in multiplexed_qubits.items():
                            qubit.readout_state(state_lin[i])
                            save(state_lin[i], state_lin_st[i])
                            with if_(state_lin[i] == 1):
                                qubit.xy.play("x180")
                        align()

                for i, qubit in multiplexed_qubits.items():
                    # hardware timestamp of the block (zero-amplitude marker)
                    qubit.xy.play("x180", amplitude_scale=0, duration=4,
                                  timestamp_stream=f"time_stamp{i + 1}")
                    save(t1_est[i], t_est_st[i])
                    save(u_inv_k[i], u_final_st[i])

        align()
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                t_est_st[i].buffer(num_blocks).save(f"t_est{i + 1}")
                u_final_st[i].buffer(num_blocks).save(f"u_final{i + 1}")
                u_evol_st[i].buffer(num_probes).save(f"u_evol{i + 1}")
                t_evol_st[i].buffer(num_probes).save(f"t_evol{i + 1}")
                state_st[i].buffer(num_blocks, num_probes).save(f"state{i + 1}")
                tau_st[i].buffer(num_blocks, num_probes).save(f"tau_ms{i + 1}")
                if interleaved:
                    state_lin_st[i].buffer(num_blocks, num_probes).save(f"state_lin{i + 1}")

    return prog, sweep_axes


def _fetch_values(results, name: str) -> np.ndarray:
    """Fetch one handle; unwrap the structured dtype timestamp streams carry."""
    handle = results.get(name)
    if handle is None:
        try:
            avail = list(results.iter_all())
        except Exception:
            avail = "unknown"
        raise RuntimeError(
            f"Bayesian T1 result handle '{name}' missing. Available: {avail}"
        )
    values = np.asarray(handle.fetch_all())
    if values.dtype.names:
        field = "value" if "value" in values.dtype.names else values.dtype.names[0]
        values = values[field]
    return np.squeeze(values)


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log: Optional[Callable] = None,
    config: Optional[dict] = None,
) -> xr.Dataset:
    """Execute and hand-fetch the heterogeneous Bayesian streams into the
    canonical dataset (T1 converted ms -> s; per-probe states/waits over
    (block_idx, probe_idx); last-block posterior evolution; timestamps ->
    elapsed block_time_s). ``lin_wait_cycles``/``interleaved`` are read back
    from the sweep_axes side-channel entries the shell attached."""
    from qualang_tools.multi_user import qm_session

    qmm = machine.connect()
    config = config if config is not None else machine.generate_config()

    qubit_names = list(np.atleast_1d(sweep_axes["qubit"].values))
    num_qubits = len(qubit_names)
    n_blocks = sweep_axes["block_idx"].values.size
    lin_wait_cycles = sweep_axes.get("lin_wait_cycles")
    interleaved = lin_wait_cycles is not None

    with qm_session(qmm, config, timeout=timeout) as qm:
        job = qm.execute(prog)
        results = job.result_handles
        results.wait_for_all_values()

        t1_s, u_final, u_evol, t1_evol = [], [], [], []
        states, taus, states_lin, stamps = [], [], [], []
        for i in range(num_qubits):
            t1_s.append(_fetch_values(results, f"t_est{i + 1}") * 1e-3)
            u_final.append(_fetch_values(results, f"u_final{i + 1}"))
            u_evol.append(_fetch_values(results, f"u_evol{i + 1}"))
            t1_evol.append(_fetch_values(results, f"t_evol{i + 1}") * 1e-3)
            states.append(_fetch_values(results, f"state{i + 1}"))
            taus.append(_fetch_values(results, f"tau_ms{i + 1}") * 1e-3)
            if interleaved:
                states_lin.append(_fetch_values(results, f"state_lin{i + 1}"))
            stamps.append(_fetch_values(results, f"time_stamp{i + 1}")[:n_blocks])

        if log:
            rep = getattr(job, "execution_report", None)
            if callable(rep):
                log(rep())
            elif rep is not None:
                log(rep)

    stamps_arr = np.asarray(stamps, dtype=float)
    block_time_s = (stamps_arr - stamps_arr[:, :1]) * 4e-9
    n_probes = np.asarray(states).shape[-1]

    data_vars = {
        "estimated_t1_s": (("qubit", "block_idx"), np.asarray(t1_s, dtype=float)),
        "u_final": (("qubit", "block_idx"), np.asarray(u_final, dtype=float)),
        "state": (("qubit", "block_idx", "probe_idx"),
                  np.asarray(states, dtype=np.int8)),
        "tau_s": (("qubit", "block_idx", "probe_idx"),
                  np.asarray(taus, dtype=float)),
        "u_evol": (("qubit", "probe_idx"), np.asarray(u_evol, dtype=float)),
        "t1_evol_s": (("qubit", "probe_idx"), np.asarray(t1_evol, dtype=float)),
        "block_time_s": (("qubit", "block_idx"), block_time_s),
    }
    if interleaved:
        data_vars["state_lin"] = (("qubit", "block_idx", "probe_idx"),
                                  np.asarray(states_lin, dtype=np.int8))
        data_vars["lin_wait_s"] = (
            ("probe_idx",),
            np.asarray(lin_wait_cycles.values, dtype=float) * 4e-9,
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "qubit": qubit_names,
            "block_idx": np.arange(n_blocks),
            "probe_idx": np.arange(n_probes),
        },
    )


from typing import Any, ClassVar

import xarray as xr

from scqo import register
from scqo.experiments import QubitT1Bayesian

from .qubit_t1_ade import discriminator_problems


def spam_terms(device, names: list[str]) -> tuple[dict, list[str]]:
    """``({target: (alpha, beta)}, problems)`` from the readout fidelities.

    alpha = P(read 0 | prep 1) = 1 - fidelity_e and beta = P(read 0 wrong way) =
    1 - fidelity_g, the two off-diagonal entries of the assignment matrix (its rows
    sum to 1). A fidelity that is unmeasured (None) or outside (0.5, 1] gives a
    message instead: at or below 0.5 the readout carries no information about that
    state, and the likelihood would divide by a SPAM span of zero or less.
    """
    spam, problems = {}, []
    for name in names:
        view = device.channel(name, "readout")
        pairs = (("fidelity_e", getattr(view, "fidelity_e", None)),
                 ("fidelity_g", getattr(view, "fidelity_g", None)))
        bad = [field for field, value in pairs
               if value is None or not (0.5 < float(value) <= 1.0)]
        if bad:
            problems.append(f"{name} has no measured {' and '.join(bad)}")
            continue
        fidelity_e, fidelity_g = (float(value) for _, value in pairs)
        spam[name] = (1.0 - fidelity_e, 1.0 - fidelity_g)
    return spam, problems


@register
class QMQubitT1Bayesian(QubitT1Bayesian):
    """Build the adaptive Bayesian T1 QUA program on the QM OPX."""

    #: Readout is held at the calibrated point for the whole run and the reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        targets = list(self.params.targets)

        spam, problems = spam_terms(self.device, targets)
        problems = discriminator_problems(machine, targets) + problems
        if problems:
            raise ValueError(
                "qubit_t1_bayesian needs a calibrated discriminator AND measured "
                "readout fidelities (the SPAM-aware likelihood reads alpha/beta "
                "from them): " + "; ".join(problems) + ". Run single_shot_readout "
                "and accept its proposals - the discriminator knobs are governed, "
                "and fidelity_g/fidelity_e are stored on every successful run."
            )

        qubits = select_qubits(machine, targets, multiplexed=False)
        lin_wait_cycles = self.lin_wait_ns() // 4

        prog, sweep_axes = build_program(
            machine,
            qubits,
            num_blocks=int(self.params.num_blocks),
            num_probes=int(self.params.num_probes),
            c_adaptive=float(self.params.adaptive_c),
            k0=float(self.params.k0),
            t1_prior_s={t: self._t1_prior_s[t] for t in targets},
            spam_pairs=spam,
            t1_min_s=float(self.params.t1_min_s),
            t1_max_s=float(self.params.t1_max_s),
            k_min=float(self.params.k_min),
            k_max=float(self.params.k_max),
            interleaved=bool(self.params.interleaved_validation),
            lin_wait_cycles=lin_wait_cycles,
            active_reset_per_probe=bool(self.params.active_reset_per_probe),
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
        )
        if self.params.interleaved_validation:
            # side-channel for the probe's acquire(): the validation grid the
            # program compiled in, so lin_wait_s lands in the dataset
            sweep_axes["lin_wait_cycles"] = xr.DataArray(lin_wait_cycles)
        return prog, sweep_axes, acquire
