"""
coordinate_transforms.py

Coordinate-frame transformations for mapping wind-tunnel test conditions
(motor angle, blade sweep/azimuthal angle, blade mounting angle) onto the
inclination angle / angle of attack seen by the blade in its own frame.

These functions were provided by a collaborator and are reproduced here
verbatim (only reformatted/typed) so the acquisition pipeline can import
them directly instead of duplicating the math.

Coordinate systems
-------------------
Global:
    x - streamwise
    y - spanwise
    z - vertical

Mount:
    x - perpendicular to mounting arm, streamwise direction for 0 deg motor 
        angle, with the mounting arm pointing in the y direction
    y - pointing from the motor to the mounting arm
    z - vertical

Swept:
    x - perpendicular to mounting arm, streamwise direction for 0 deg motor 
        angle, with the mounting arm pointing in the y direction
    y - pointing from the motor to the mounting arm
    z - pointing up the blade from tip to root

All angle arguments/returns for the low-level functions are in radians,
matching the collaborator's original notebook. Degree-based convenience
wrappers are provided at the bottom of this file.


Angle conventions
-----------------
Motor angle: 0 deg is aligned with the global x axis, positive is clockwise
Mounting angle: AoA when motor angle = 0 deg
Sweep angle: 0 deg is vertical, negative is rotation about the mount x axis


"""

import numpy as np


# --------------------------------------------------
# Low-level transforms (radians in, radians out)
# --------------------------------------------------


def global_to_mount(U, motor_angle):
    """Convert global wind components to mount-aligned components based
    on the motor angle."""
    u_g, v_g, w_g = U
    R = np.array([
        [np.cos(motor_angle), -np.sin(motor_angle), 0],
        [np.sin(motor_angle), np.cos(motor_angle), 0],
        [0, 0, 1],
    ])
    U_mount = R @ np.array([u_g, v_g, w_g])
    return U_mount


def mount_to_swept(U, sweep_angle):
    """Convert mount-aligned wind components to swept components
    based on the sweep angle (rotation about the mount x-axis)."""
    u_m, v_m, w_m = U
    rot_angle = sweep_angle
    R = np.array([
        [1, 0, 0],
        [0, np.cos(rot_angle), -np.sin(rot_angle)],
        [0, np.sin(rot_angle), np.cos(rot_angle)],
    ])
    U_swept = R @ np.array([u_m, v_m, w_m])
    return U_swept


def inclination_from_swept(U):
    """Inclination angle of the wind relative to the blade, from the
    swept wind components. Tip-to-root is positive."""
    u_b, v_b, w_b = U
    horizontal_speed = np.sqrt(u_b ** 2 + v_b ** 2)
    # Negative sign because positive w_b is up the blade, but positive
    # inclination is defined as pointing down towards the root.
    inclination_angle = np.arctan2(w_b, horizontal_speed)
    return inclination_angle


def angle_of_attack_from_swept(U, mounting_angle):
    """Angle of attack of the wind relative to the blade chord line, from
    the swept wind components and the (mounting/blade) pitch
    angle. `pitch` must be in radians, consistent with the other angles
    here."""
    u_b, v_b, w_b = U
    # wind_angle = np.arctan2(u_b, -v_b)
    wind_angle = np.arctan2(v_b, u_b)
    angle_of_attack = wind_angle + mounting_angle
    return angle_of_attack


# --------------------------------------------------
# Convenience wrapper (degrees in, degrees out)
# --------------------------------------------------


def _wrap_deg(angle_deg):
    """Wrap an angle in degrees into (-180, 180]."""
    return (angle_deg + 180) % 360 - 180


def compute_frame_angles(motor_angle_deg, sweep_angle_deg, mounting_angle_deg,
                          U_global=(1.0, 0.0, 0.0)):
    """Map a (motor_angle, sweep_angle, mounting_angle) test condition to
    the resulting inclination angle and angle of attack seen by the blade.

    Mapping (per the collaborator's notebook):
        yaw              <- motor_angle      (the motor sweeps effective
                                                wind yaw relative to the rig)
        azimuthal_angle  <- sweep_angle       (fixed blade sweep/orientation
                                                for this run)
        pitch            <- mounting_angle    (fixed blade mounting angle
                                                for this run)

    All *_deg inputs are in degrees. Returns a dict of both the raw
    (unwrapped) and shifted (wrapped to -180..180 deg) inclination angle
    and angle of attack, plus the radian versions used in the calculation,
    so everything needed for a log file is available in one place.
    """
    motor_angle_rad = np.radians(motor_angle_deg)
    sweep_angle_rad = np.radians(sweep_angle_deg)
    mounting_angle_rad = np.radians(mounting_angle_deg)

    U_mount = global_to_mount(np.asarray(U_global, dtype=float), motor_angle_rad)
    U_swept = mount_to_swept(U_mount, sweep_angle_rad)

    inclination_angle = inclination_from_swept(U_swept)
    angle_of_attack = angle_of_attack_from_swept(U_swept, mounting_angle_rad)

    inclination_angle_deg = np.degrees(inclination_angle)
    angle_of_attack_deg = np.degrees(angle_of_attack)

    return {
        "yaw_angle_deg": motor_angle_deg,
        "azimuthal_angle_deg": sweep_angle_deg,
        "mounting_angle_deg": mounting_angle_deg,
        "inclination_angle_deg": inclination_angle_deg,
        "inclination_angle_deg_shifted": _wrap_deg(inclination_angle_deg),
        "angle_of_attack_deg": angle_of_attack_deg,
        "angle_of_attack_deg_shifted": _wrap_deg(angle_of_attack_deg),
    }
