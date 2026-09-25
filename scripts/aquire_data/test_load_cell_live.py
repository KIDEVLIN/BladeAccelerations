#!/usr/bin/env python3
"""
test_load_cell_live.py

Standalone diagnostic: continuously streams raw AI0-AI5 voltages from the
load cell's NI DAQ channels, plus the calibrated Fx/Fy/Fz/Mx/My/Mz, so you
can watch the numbers live while you tap or load the cell by hand.

This bypasses collect_load_for_duration entirely (no CSV, no barrier/sync,
no fixed duration) -- it's just meant to answer one question: does ANYTHING
change on these channels when the cell is loaded?

Usage:
    python test_load_cell_live.py --device Dev11
    python test_load_cell_live.py --device Dev11 --rse   # try single-ended
                                                            # if DIFF looks dead

Ctrl+C to stop.
"""

import argparse
import time

import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration


# Reuse the real calibration so the "does it move" check also tells you
# whether the numbers are in a sane force/moment range.
from load_cell import CALIBRATION_MATRIX, ZERO_OFFSET, OUTPUT_LABELS

N_CHANNELS = 6
SAMPLE_RATE_HZ = 1000.0
BATCH_SAMPLES = 200  # ~0.2s per refresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="Dev11")
    parser.add_argument("--rse", action="store_true",
                         help="Use RSE (single-ended) instead of DIFF, "
                              "in case the wiring isn't actually differential")
    args = parser.parse_args()

    term_cfg = TerminalConfiguration.RSE if args.rse else TerminalConfiguration.DIFF
    print(f"Device: {args.device}   Terminal config: {'RSE' if args.rse else 'DIFF'}")
    print("Streaming raw volts + calibrated force/moment. Ctrl+C to stop.\n")

    with nidaqmx.Task() as task:
        for ch in range(N_CHANNELS):
            task.ai_channels.add_ai_voltage_chan(
                f"{args.device}/ai{ch}",
                terminal_config=term_cfg,
                min_val=-10.0,
                max_val=10.0,
            )
        task.timing.cfg_samp_clk_timing(
            SAMPLE_RATE_HZ,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=BATCH_SAMPLES * 4,

        )
        task.start()

        baseline = None
        try:
            while True:
                raw = task.read(number_of_samples_per_channel=BATCH_SAMPLES,
                                 timeout=5.0)
                data = np.array(raw)  # (6, BATCH_SAMPLES)
                means = data.mean(axis=1)
                ptp = data.ptp(axis=1)  # peak-to-peak within this batch -- shows noise/activity

                if baseline is None:
                    baseline = means.copy()

                delta = means - baseline

                F = (CALIBRATION_MATRIX @ means.reshape(6, 1)) - ZERO_OFFSET

                stg_str = "  ".join(
                    f"stg{i}:{means[i]:+.5f}V(pp{ptp[i]:.5f},d{delta[i]:+.5f})"
                    for i in range(N_CHANNELS)
                )
                force_str = "  ".join(
                    f"{label}:{val[0]:+8.3f}" for label, val in zip(OUTPUT_LABELS, F)
                )

                print(f"\r{stg_str}", flush=True)
                print(f" {force_str}   ", end="\r\033[1A", flush=True)

        except KeyboardInterrupt:
            print("\n\nStopped.")


if __name__ == "__main__":
    main()