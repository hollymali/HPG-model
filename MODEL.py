"""Shared Meeting 15 model with sinusoidal SCN.

Time is in hours from 08:00. The SCN ranges from scn_min to scn_max.
The default scn_phase of 16.65 hours puts its peak at 00:39.
The sleep multiplier still applies separately from 23:00 to 07:00.
The same Euler updates serve full traces, sampled sweeps and added-E2 experiments.
"""

import numpy as np


"""These are our parameter constants, our simple model sets as many to 1 as possible"""
REFERENCE = {
    "constant_inflow": 6, # Basal E2 production, pg/mL
    "production_scale": 210.7, # Maximum LH-based production, pg/mL
    "activation_midpoint": 1, # Half saturation density of LH for stimulation of E2
    "activation_exponent": 5, # Hill coefficient for LH stimulation of E2
    "inhibition_midpoint": 1, # Half saturation density of LH for inhibition of ovarian responsiveness
    "inhibition_exponent": 5, # Hill coefficient for LH inhibition of ovarian responsiveness
    "n_gain": 1, # Maximum E2 modulated production of N
    "n_e2_midpoint": 30.2, # Half saturation density of E2 for inhibition of N production
    "e2_feedback_exponent": 2.36, # Hill coefficient for E2 inhibition of N production
    "n_period_exponent": 2.36, # Hill coefficient for N modulated decrease in GnRH interpulse period
    "n_amplitude_exponent": 2.36, # Hill coefficient for N modulated increase in GnRH pulse amplitude
    "n_relaxation": 1, # Constant decay term for N
    "gnrh_pulse_jump": 1, # Maximum GnRH pulse amplitude, modulated by N
    "lh_gnrh_gain": 1, # Maximum GnRH activation of LH production
    "period_min": 67.5, # Slowest GnRH interpulse period (based on )
    "period_max": 76.73, # Longest GnRH interpulse period period (based on )
    "scn_min": 35, # Minimum SCN value
    "scn_max": 277, # Maximum SCN value
    "sleep_multiplier": 1.55, # GnRH interpulse period multiplier during sleep
    "scn_phase": 16.65, # Peak of SCN effect on amplitude, equivalent to clock time 00:39 from 08:00 (based on SCN-dependent LH surge timing, )
    "gnrh_half_life_min": 4.09, # Half life of GnRH in minutes (based on )
    "lh_half_life_min": 29, # Half life of LH in minutes (based on)
    "e2_half_life_min": 22.53, # Half life of E2 in minutes (based on )
}
KEYS = tuple(REFERENCE)
START_CLOCK_H = 8
STEP_H = 1.5 / 3600. # Solver uses 1.5 second steps, which is equivalent to 1.5/3600 hours
WARMUP_H = 96. # Simulate for 4 days to let system settle
OUTPUT_H = 72. # Simulate for 3 days after warmup to measure behavior


"""Default parameters match the REFERENCE dict"""
def simulate_endocrine(
    output_h=OUTPUT_H,
    warmup_h=WARMUP_H,
    step_h=STEP_H,
    constant_inflow=6,
    production_scale=210.7,
    activation_midpoint=1,
    activation_exponent=5,
    inhibition_midpoint=1,
    inhibition_exponent=5,
    n_gain=1,
    n_e2_midpoint=30.2,
    e2_feedback_exponent=2.36,
    n_period_exponent=2.36,
    n_amplitude_exponent=2.36,
    n_relaxation=1,
    gnrh_pulse_jump=1,
    lh_gnrh_gain=1,
    period_min=67.5,
    period_max=76.73,
    scn_min=35,
    scn_max=277,
    sleep_multiplier=1.55,
    scn_phase=16.65,
    gnrh_half_life_min=4.09,
    lh_half_life_min=29,
    e2_half_life_min=22.53,
    added_e2=0,
    for_fft=False,
):
    """Integrate with independent E2 feedback, pulse-period and amplitude powers.

    E2 feedback also determines initial N. The two N pulse maps use only their
    respective exponents. Setting all three to h reproduces the old shared h.
    M_sleep(t) is sleep_multiplier during 23:00-07:00 and 1 while awake.
    The sleep_indicator output is retained for existing plotting callers.

    Normal mode returns all model traces, including warmup, at every Euler step.
    for_fft=True returns only ovarian E2, sampled every seven minutes after warmup
    over output_h hours (excluding the endpoint). Time remains the last axis.
    added_e2 affects N feedback; E2 remains ovarian E2 in the returned trace.
    """
    # Create array with 0->number of total hours using given step size.
    end = round((warmup_h + output_h) / step_h)
    time_h = np.arange(end + 1) * step_h
    shape = np.broadcast(scn_min, sleep_multiplier, activation_exponent,
                         inhibition_exponent, e2_feedback_exponent,
                         n_period_exponent, n_amplitude_exponent, added_e2).shape
    if for_fft:
        fft_e2 = []
        start = round(warmup_h / step_h)
        every = round((7 / 60) / step_h)  # 280 steps at the default 1.5 s step.

    # Create array with hours converted to clock time (ZT).
    clock_h = (time_h + START_CLOCK_H) % 24
    # Create sleep boolean mask where Licinio subjects slept.
    sleep = (clock_h >= 23) | (clock_h < 7)
    # Initialize values, one independent state per parameter setting.
    gnrh, lh, e2, phi = (np.full(shape, value) for value in (0., 3., 19., 0.))
    n_state = (
        n_gain*n_e2_midpoint**e2_feedback_exponent
        / (n_e2_midpoint**e2_feedback_exponent + (e2+added_e2)**e2_feedback_exponent)
    )
    responsiveness = inhibition_midpoint**inhibition_exponent/(inhibition_midpoint**inhibition_exponent+lh**inhibition_exponent)
    gnrh_decay = 60.*np.log(2.)/gnrh_half_life_min
    lh_decay = 60.*np.log(2.)/lh_half_life_min
    e2_decay = 60.*np.log(2.)/e2_half_life_min

    if not for_fft:
        # Full traces for normal graph output, including the initial state.
        n_trace, gnrh_trace, lh_trace, e2_trace, phi_trace, r_trace = (
            np.empty((*shape, len(time_h))) for _ in range(6)
        )
        n_trace[..., 0], gnrh_trace[..., 0], lh_trace[..., 0] = n_state, gnrh, lh
        e2_trace[..., 0], phi_trace[..., 0], r_trace[..., 0] = e2, phi, responsiveness

    # Programmatically perform Euler's method.
    for i in range(len(time_h) - 1):
        # Compute the sinusoidal SCN value at this time.
        scn = (scn_max + scn_min)/2 + (scn_max - scn_min)/2 * np.cos(
            2*np.pi*(time_h[i] % 24 - scn_phase)/24)
        # Use the sleep multiplier if sleep is True, and 1 otherwise.
        m_sleep = sleep_multiplier if sleep[i] else 1.
        if for_fft and start <= i < end and (i - start) % every == 0:
            fft_e2.append(e2)

        # Every update uses the state at the start of this Euler step.
        # Phi is the pulse timer: it crosses an integer faster when N is larger,
        # because a smaller period means a larger step towards the next integer.
        next_phi = phi+60.*step_h/(
            (period_max - (period_max-period_min)*n_state**n_period_exponent
             / (1. + n_state**n_period_exponent)) * m_sleep)
        next_n = n_state+step_h*n_relaxation*(
            n_gain*n_e2_midpoint**e2_feedback_exponent
            / (n_e2_midpoint**e2_feedback_exponent + (e2+added_e2)**e2_feedback_exponent)
            - n_state)
        next_gnrh = gnrh-step_h*gnrh_decay*gnrh+(
            gnrh_pulse_jump*n_state**n_amplitude_exponent / (1. + n_state**n_amplitude_exponent)
            *scn*(np.floor(next_phi)-np.floor(phi)))
        next_lh = lh+step_h*(lh_gnrh_gain*gnrh-lh_decay*lh)
        next_r = responsiveness+step_h*(
            inhibition_midpoint**inhibition_exponent/(inhibition_midpoint**inhibition_exponent+lh**inhibition_exponent)-responsiveness)
        next_e2 = e2+step_h*(constant_inflow+production_scale
            *lh**activation_exponent/(activation_midpoint**activation_exponent+lh**activation_exponent)
            *responsiveness-e2_decay*e2)
        phi, n_state, gnrh, lh = next_phi, next_n, next_gnrh, next_lh
        responsiveness, e2 = next_r, next_e2
        if not for_fft:
            n_trace[..., i + 1], gnrh_trace[..., i + 1], lh_trace[..., i + 1] = n_state, gnrh, lh
            e2_trace[..., i + 1], phi_trace[..., i + 1], r_trace[..., i + 1] = e2, phi, responsiveness
    if for_fft:
        return np.stack(fft_e2, axis=-1)

    scn_trace = (scn_max + scn_min)/2 + (scn_max - scn_min)/2 * np.cos(
        2*np.pi*(time_h % 24 - scn_phase)/24)
    return dict(time_h=time_h, N=n_trace, GnRH=gnrh_trace, LH=lh_trace,
                E2=e2_trace, phi=phi_trace, R=r_trace,
                SCN=scn_trace, sleep_indicator=sleep.astype(float))

