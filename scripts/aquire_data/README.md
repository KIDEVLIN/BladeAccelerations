# scripts/aquire_data

Acquisition code for the lab PC, and the post-processing that turns run
folders into summaries and plots. Run the scripts from this folder, so that
`utils/` is importable. Each script's docstring gives its options.

## Acquisition (lab PC, needs the hardware libraries)

| script | does |
|---|---|
| `main.py` | Runs one angle sweep: moves the motor, captures accelerometers, load cell and tunnel conditions at each angle, and writes the run folder. The fixed rig constants (`SWEEP_ANGLE_DEG`, `CHORD_OFFSET_DEG` → `MOUNTING_ANGLE_DEG`) are set at the top |
| `ring_down_aquire.py` | Free-decay (tip-flick) tests for natural frequencies and damping |
| `motor.py` | Servo drive and encoder. Motor angle = −counts / 694.44; positive turns the arm clockwise seen from above |
| `load_cell.py` | Synchronized NI-DAQ load-cell capture |
| `wind_tunnel_conditions.py` | Tunnel pressure, temperature and velocity; port of the LabVIEW VIs |
| `test_load_cell_live.py`, `test_modbus.py`, `check_velocity.py` | Bench diagnostics |

## Post-processing (needs only the run folders)

| script | does |
|---|---|
| `extract_run_summary.py` | **Start here.** Reads each run once and writes `angle_summary.csv`, `spectra.npz` and `run_info.json` per run, plus the `master_*.csv` tables |
| `angle_metrics.py` | Run-level vibration metrics and per-region flags (R1-R3) from the summaries |
| `plot_angle_sweep.py` | Normalized variance against motor angle |
| `plot_psd.py` | Spectra: angle maps, single angles, and runs compared |
| `plot_design_space.py` | One point per run on Reynolds number against Cauchy number |
| `plot_regions_inflow.py` | Where the vibration regions fall in inclination and angle of attack |
| `plot_summary.py` | Quick look at one run from its summary files |
| `plot_variance_sweep.py`, `plot_variance_compare.py`, `postprocessing_pipeline.py` | Variance diagnostics of one or several runs, read from the raw folders |
| `normalization.py`, `variance_analysis.py` | Shared metric definitions |
| `explore.py` | Interactive scratchpad (`# %%` cells) |

## utils/

| module | does |
|---|---|
| `coordinate_transforms.py` | Motor, sweep and mounting angle → inclination and angle of attack. Conventions are in its docstring; the checks are in `reports/angle_check` |
| `run_loader.py` | Loads a run folder; encoder-window motor angle |
| `live_variance_plot.py` | The live plot `main.py` shows during a run |
| `plot_timeseries.py` | Time-series figure for one angle capture |

## tests/

```bash
poetry run python -m unittest discover -s scripts/aquire_data/tests   # from the repo root
```

- `test_coordinate_transforms.py` checks the transform against the
  experimentalists' sketch, an independent vector construction, the original
  implementation and a hand calculation.
- `test_angle_chain.py` checks every link from encoder counts to the summary
  tables on the recorded data. These tests are skipped if `data/` is absent.
- `render_rig.py` and `render_parked_turbine.py` draw the rig, or a parked
  turbine, at a given angle, with the angles computed from vectors.
- `plot_test_matrix.py` plots the test matrix in inclination and angle of
  attack.

## Run folder layout (written by main.py)

```
<run>/
  run_metadata.json             rig constants, tunnel setup, per-angle records
  coordinate_frame_angles.csv   per angle: requested and measured motor angle, flow angles, sync times
  angle_+120.00deg.csv          accelerometers: 4 sensors x 3 axes, raw int16 (header: ODR, mg/LSB, start time)
  angle_+120.00deg_load.csv     load cell: 6 gauges, Fx Fy Fz Mx My Mz, 2 kHz
  encoder_log.txt               motor encoder counts (header: start time on the same clock)
  tunnel_conditions_summary.csv
```

The accelerometer files carry the configured ODR (1600 Hz), but the true
rate is 1493 Hz, with gaps between reads. Use
`scripts/data_analysis/accel_spectra.py`, not the header rate. If an angle was
measured twice in a run, its file holds the second capture.
