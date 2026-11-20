"""
coordinate_transforms.py

Coordinate-frame transformations for mapping wind-tunnel test conditions
(motor angle, blade sweep angle, blade mounting angle) onto the inclination
angle and angle of attack seen by the blade.

The geometry follows the experimentalists' sketch of the rig (29 Sep 2026)
and was checked against an independent vector construction and against the
collaborator's original implementation (origin/master), which it matches
for horizontal tunnel flow. See reports/angle_check/ and
tests/test_coordinate_transforms.py.

The rig
-------
A motor turns a vertical shaft. A horizontal arm sticks out from the shaft,
and the blade hangs from the arm tip: its root is at the arm, its tip below.
The blade leans by the sweep angle within the vertical plane of the arm,
tip toward the shaft for the production runs (sweep -30 deg). Its chord is
(nearly) perpendicular to the arm.

Coordinate systems
------------------
Global:
    x - streamwise (the tunnel flow is +x)
    y - horizontal, to the left looking downstream
    z - vertical, up
Mount (turns with the motor):
    x - horizontal, perpendicular to the arm; = global x at motor angle 0
    y - along the arm, from the shaft to the arm tip
    z - vertical, up
Swept (fixed to the blade):
    x - = mount x
    y - in the blade's cross-section plane, toward the arm-tip side
    z - along the span, from tip to root

Angle conventions
-----------------
Motor angle: 0 deg puts the arm across the flow (arm along global +y);
    positive turns the arm clockwise seen from above, so +90 deg points the
    arm downstream and -90 deg upstream.
Sweep angle: rotation of the swept frame from the mount frame about mount
    x. 0 = blade vertical. Negative tilts the tip-to-root axis toward the
    arm tip, i.e. the blade tip leans back toward the shaft.
Mounting angle: angle between the chord line and the arm, in the blade's
    cross-section plane. 90 deg = chord perpendicular to the arm, which at
    motor angle 0 puts the chord along the flow (leading edge upstream).
    The angle of attack at motor 0 is therefore 90 - mounting angle.
Inclination: angle of the flow out of the blade's cross-section plane;
    positive when the flow runs along the span from tip to root.
Angle of attack: angle of the flow, projected into the cross-section plane,
    from the chord line (leading edge -> trailing edge); positive toward
    the arm-tip side, so it grows with the motor angle.

History: the version of 1 Oct 2026 (commit 2965360) rotated the wind vector
by the sweep angle instead of the frame (flipping the sign of every
inclination) and added the mounting angle to the angle of attack directly
(treating 84.105 deg as the angle of attack at motor 0, i.e. the chord
nearly along the arm). Summaries written with it carry those errors.

All angle arguments/returns for the low-level functions are in radians.
compute_frame_angles() takes and returns degrees.
"""

import numpy as np


# --------------------------------------------------
# Low-level transforms (radians in, radians out)
# --------------------------------------------------


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def global_to_mount(U, motor_angle):
    """Global wind components -> mount components. The mount frame is the
    global frame turned by `motor_angle` clockwise seen from above (a
    rotation of -motor_angle about z), so its components are
    R_z(-motor_angle)^T U = R_z(motor_angle) U."""
    return _rot_z(motor_angle) @ np.asarray(U, dtype=float)


def mount_to_swept(U, sweep_angle):
    """Mount wind components -> swept (blade) components. The swept frame
    is the mount frame rotated by `sweep_angle` about mount x, so its
    components are R_x(sweep_angle)^T U."""
    return _rot_x(sweep_angle).T @ np.asarray(U, dtype=float)


def inclination_from_swept(U):
    """Inclination of the wind relative to the blade, from the swept
    components: the angle out of the cross-section plane, positive when the
    wind runs from tip to root (swept z points from tip to root)."""
    u_s, v_s, w_s = U
    return np.arctan2(w_s, np.hypot(u_s, v_s))


def angle_of_attack_from_swept(U, mounting_angle):
    """Angle of attack from the swept components. The chord makes
    `mounting_angle` with the arm (swept y), so it lies at
    pi/2 - mounting_angle from swept x toward -y; the wind's in-plane angle
    from swept x is atan2(v, u)."""
    u_s, v_s, w_s = U
    return np.arctan2(v_s, u_s) + (np.pi / 2 - mounting_angle)


# --------------------------------------------------
# Convenience wrapper (degrees in, degrees out)
# --------------------------------------------------


def _wrap_deg(angle_deg):
    """Wrap an angle in degrees into [-180, 180)."""
    return (angle_deg + 180) % 360 - 180


def compute_frame_angles(motor_angle_deg, sweep_angle_deg, mounting_angle_deg,
                         U_global=(1.0, 0.0, 0.0)):
    """Inclination and angle of attack seen by the blade for a test
    condition (motor angle, sweep angle, mounting angle; degrees). Returns
    the raw and wrapped ([-180, 180)) angles in degrees. The yaw/azimuthal
    keys repeat the inputs under the collaborator's original names."""
    U_mount = global_to_mount(U_global, np.radians(motor_angle_deg))
    U_swept = mount_to_swept(U_mount, np.radians(sweep_angle_deg))

    inclination_angle_deg = np.degrees(inclination_from_swept(U_swept))
    angle_of_attack_deg = np.degrees(
        angle_of_attack_from_swept(U_swept, np.radians(mounting_angle_deg)))

    return {
        "yaw_angle_deg": motor_angle_deg,
        "azimuthal_angle_deg": sweep_angle_deg,
        "mounting_angle_deg": mounting_angle_deg,
        "inclination_angle_deg": inclination_angle_deg,
        "inclination_angle_deg_shifted": _wrap_deg(inclination_angle_deg),
        "angle_of_attack_deg": angle_of_attack_deg,
        "angle_of_attack_deg_shifted": _wrap_deg(angle_of_attack_deg),
    }
