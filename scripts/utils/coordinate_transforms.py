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

Turbine:
    x - pointing into turbine (down nacelle)
    y - pointing to left on turbine seen from in front
    z - vertical

Blade:
    x - pointing into rotor plane
    y - pointing to negative tangential direction in rotor plane
    z - pointing up the blade (root to tip)

All angle arguments/returns for the low-level functions are in radians,
matching the collaborator's original notebook. Degree-based convenience
wrappers are provided at the bottom of this file.
"""

import numpy as np


# --------------------------------------------------
# Low-level transforms (radians in, radians out)
# --------------------------------------------------


def global_to_turbine(U, yaw):
    """Convert global wind components to turbine-aligned components based
    on the yaw angle (rotation about the global z-axis)."""
    u_g, v_g, w_g = U
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    U_turbine = R @ np.array([u_g, v_g, w_g])
    return U_turbine


def turbine_to_blade(U, azimuthal_angle):
    """Convert turbine-aligned wind components to blade-aligned components
    based on the azimuthal angle (rotation about the turbine x-axis)."""
    u_t, v_t, w_t = U
    rot_angle = azimuthal_angle
    R = np.array([
        [1, 0, 0],
        [0, np.cos(rot_angle), -np.sin(rot_angle)],
        [0, np.sin(rot_angle), np.cos(rot_angle)],
    ])
    U_blade = R @ np.array([u_t, v_t, w_t])
    return U_blade


def inclination_from_blade(U):
    """Inclination angle of the wind relative to the blade, from the
    blade-aligned wind components. Tip-to-root is positive."""
    u_b, v_b, w_b = U
    horizontal_speed = np.sqrt(u_b ** 2 + v_b ** 2)
    # Negative sign because positive w_b is up the blade, but positive
    # inclination is defined as pointing down towards the root.
    inclination_angle = np.arctan2(-w_b, horizontal_speed)
    return inclination_angle


def angle_of_attack_from_blade(U, pitch):
    """Angle of attack of the wind relative to the blade chord line, from
    the blade-aligned wind components and the (mounting/blade) pitch
    angle. `pitch` must be in radians, consistent with the other angles
    here."""
    u_b, v_b, w_b = U
    wind_angle = np.arctan2(u_b, -v_b)
    angle_of_attack = wind_angle - pitch
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
    yaw = np.radians(motor_angle_deg)
    azimuthal_angle = np.radians(sweep_angle_deg)
    pitch = np.radians(mounting_angle_deg)

    U_turbine = global_to_turbine(np.asarray(U_global, dtype=float), yaw)
    U_blade = turbine_to_blade(U_turbine, azimuthal_angle)

    inclination_angle = inclination_from_blade(U_blade)
    angle_of_attack = angle_of_attack_from_blade(U_blade, pitch)

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
