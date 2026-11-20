# Rig angle mapping: motor angle → inflow on the blade

Use this to set the inflow direction in a simulation of the DTU10MW tunnel
blade so that it matches the experiment. The values were checked on
12 Nov 2026. `angle_check.pdf` in this folder has the evidence (34 tests,
including against the angles recorded live during the runs). The code is
`scripts/aquire_data/utils/coordinate_transforms.py` in BladeAccelerations.

## 1. The rig

- A motor turns a vertical shaft, and a horizontal arm sticks out from it.
- The blade **hangs from the arm tip**: the root is at the arm and the tip is
  below.
- The blade leans 30° from vertical in the vertical plane that contains the
  arm, with its **tip back toward the shaft** (sweep β = −30°).
- The chord is **5.895° off perpendicular to the arm**. The mounting angle is
  μ = 90 − 5.895 = 84.105°, the chord-to-arm angle.
- The **leading edge is upstream at motor 0°**.
- The tunnel flow is horizontal, along global +x. Global z points up.
- **Motor angle θ:** 0° puts the arm across the flow. Positive turns the arm
  **clockwise seen from above**, so +90° points the arm downstream and −90°
  upstream.

All 32 production runs use β = −30° and μ = 84.105°. Only θ changes within a
run.

## 2. Angle definitions

All vectors are unit vectors, defined on the blade itself. The simulation needs
three of them:

- **ŝ**: the span direction, **root → tip**.
- **ĉ**: the chord line of the reference section, **leading edge → trailing
  edge**. The 5.895° is measured on this section (see §6, item 1).
- **n̂ = ĉ × ŝ**: the section normal. At motor 0° it points to the arm-tip
  side, away from the shaft.

With Û the freestream direction:

- **Inclination** i = asin(−Û·ŝ). It is positive when the flow runs along the
  span **from tip to root**.
- **Angle of attack** α = atan2(Û·n̂, Û·ĉ). It is the in-plane flow angle
  measured from the chord. α = 0 means the flow runs leading edge → trailing
  edge; ±180° means trailing edge → leading edge. It increases with θ.

Inverse, which is what a simulation needs: the freestream direction in blade
axes is

```
U_hat = cos(i)·cos(α)·ĉ + cos(i)·sin(α)·n̂ − sin(i)·ŝ
```

**n̂ points out of the suction side.** The beam model's frame
(`prototype-blade-exawind`, `docs/beam_model/report.pdf`, "Visual checks in
3D") has X toward the suction side, Y toward the trailing edge and Z along the
span, root → tip. So ĉ ≈ +Y, ŝ = +Z, and n̂ = ĉ × ŝ = Y × Z = +X, the suction
side. A **positive angle of attack therefore means the flow strikes the
pressure side**, the usual airfoil convention. Two things follow:

- With the rig assumptions of §1, the **suction side faces away from the
  shaft at motor 0°**. Anyone at the rig can check this by eye.
- In BeamDyn axes (X, Y, Z), Û = (Û·n̂, Û·ĉ, Û·ŝ) when the reference chord
  lies along +Y. Otherwise, first rotate about Z by the reference section's
  chord angle from +Y. Still compute n̂ = ĉ × ŝ on the actual mesh.

## 3. Closed form (horizontal flow)

```
i = asin(−sin β · sin θ)                     # = asin(0.5 sin θ) for β = −30°
α = atan2(cos β · sin θ, cos θ) + (90° − μ)  # 90° − μ = +5.895°
```

Both angles are wrapped to [−180°, 180°). `coordinate_frame_angles.csv` in
the raw run folders stores the **unwrapped** α as well; for example −230.4° at
120° is the same angle as +129.6°.

## 4. Check values (β = −30°, μ = 84.105°)

| motor θ | inclination i | angle of attack α | Û·ĉ | Û·n̂ | Û·ŝ |
|---|---|---|---|---|---|
| 0° | 0.000° | +5.895° | +0.9947 | +0.1027 | 0.0000 |
| +90° | +30.000° | +95.895° | −0.0889 | +0.8614 | −0.5000 |
| −90° | −30.000° | −84.105° | +0.0889 | −0.8614 | +0.5000 |
| **+120°** | **+25.659°** | **+129.585°** | **−0.5744** | **+0.6947** | **−0.4330** |
| 180° | 0.000° | −174.105° | −0.9947 | −0.1027 | 0.0000 |

Any implementation should reproduce this table.

## 5. The 120° case in tunnel coordinates

For run `DTU10MW_385psi_800fs` at motor 120°, if you build the domain in the
tunnel frame (x downstream, z up, flow along +x), place the blade with:

- arm direction (shaft → arm tip): ( 0.8660, −0.5000,  0.0000)
- ŝ, root → tip:                   (−0.4330,  0.2500, −0.8660)
- ĉ, LE → TE:                      (−0.5744, −0.8170,  0.0514)
- n̂ = ĉ × ŝ:                       ( 0.6947, −0.5197, −0.4974)

The flow is along (1, 0, 0). These vectors reproduce i = +25.659° and
α = +129.585°. If the blade is kept in its own mesh frame instead, set the
inflow to Û = −0.5744 ĉ + 0.6947 n̂ − 0.4330 ŝ. In BeamDyn axes, with the
reference chord along +Y, that is **Û = (+0.6947, −0.5744, −0.4330)**: the
flow comes from the pressure side, travels from the trailing edge toward the
leading edge, and runs from tip to root.

Flow conditions of that run (`data/Production_summaries/DTU10MW_385psi_800fs/run_info.json`):

| quantity | value |
|---|---|
| U | 6.907 m/s |
| ρ | 32.35 kg/m³ |
| μ_air | 1.880e−5 Pa·s |
| T | 25.7 °C |
| p_abs | 2.758 MPa (385 psi gauge) |
| Re (mean chord 13.8 mm) | 1.64e5 |
| span | 0.28 m |

## 6. Assumptions to state in the simulation setup

These rest on the experimentalists' sketch (29 Sep 2026) and on the
acquisition code, not on any test:

1. **The reference section for the 5.895°.** It is not recorded which
   section's chord was used. Assume the root (mounting) section unless Kyle
   Devlin says otherwise. Because the blade is twisted, using the chord of a
   different station shifts α by the twist difference.
2. **The direction of the 5.895° turn.** It was entered so that α = +5.895° at
   motor 0°.
3. **The blade hangs root-up and tip-down**, with the leading edge upstream at
   motor 0°. If the leading edge actually faced downstream, every α would
   change by 180°. Together with the blade's geometry, this puts the
   suction side facing away from the shaft at motor 0°, which can be checked
   by eye (§2).
4. **The motor turns clockwise from above for positive θ.** This comes from
   the sketch and the 29 Sep commit "Correcting rotation direction".

## 7. Do not use

- **`angle_summary.csv` / `master_angles.csv` written before 12 Nov 2026.**
  They were made with the 1 Oct transform, commit `2965360`. In those files
  the inclination sign is flipped and α is 78.21° too high; for example the
  120° case reads −25.7° / −152.2°. They have been regenerated. Check that the
  120° row of `DTU10MW_385psi_800fs` reads +25.659° / +129.585°.
- **"Mounting angle" as the angle of attack at motor 0.** It is the
  chord-to-arm angle. The angle of attack at motor 0 is 90° − μ.
- **Turbine-equivalent mappings** in `tests/plot_test_matrix_vs_turbine.py`
  and `tests/map_angles.py`. They have not been validated.

## 8. Reference implementation

```python
from utils import coordinate_transforms as ct   # scripts/aquire_data on sys.path
r = ct.compute_frame_angles(motor_angle_deg=120.0, sweep_angle_deg=-30.0,
                            mounting_angle_deg=84.105)
r["inclination_angle_deg_shifted"], r["angle_of_attack_deg_shifted"]   # 25.659, 129.585
```

To validate it, run
`poetry run python -m unittest discover -s scripts/aquire_data/tests`
from the BladeAccelerations root. It should report 34 tests OK.
