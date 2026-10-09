"""
normalization.py

Nondimensionalisation of the accelerometer variance so runs taken at different
wind tunnel conditions / blades / materials can be compared.

Characteristic acceleration (force scale / mass):

    a_c = F_c / m,    F_c = (1/2 rho U^2) * (c_avg * b)

      rho    air density (kg/m^3)               -- run-averaged tunnel conditions
      U      free-stream velocity (m/s)         -- run-averaged tunnel conditions
      c_avg  average chord along the blade (m)
      b      blade span (m)
      m      blade mass (kg)

    a* = a / a_c        variance plotted as a*^2 = Var(a) / a_c^2   (dimensionless)

The accelerometer variance is handled in g^2, so a_c is converted to g
(a_c_g = a_c / 9.80665) and a*^2 = Var_g2 / a_c_g^2.

Inputs come from each run's run_metadata.json:
    tunnel_conditions.density_kg_m3, tunnel_conditions.velocity_freestream_m_s
    blade_chord_avg_m, blade_span_m, blade_mass_kg
(main.py records the blade values from its BLADE_* constants.) Any of them can be
overridden per run from the post-processing command line.
"""

import math

G = 9.80665  # m/s^2

# name -> (metadata lookup path, unit label); used for messages and lookups
INPUTS = {
    "density_kg_m3": (("tunnel_conditions", "density_kg_m3"), "kg/m^3"),
    "velocity_m_s": (("tunnel_conditions", "velocity_freestream_m_s"), "m/s"),
    "chord_avg_m": (("blade_chord_avg_m",), "m"),
    "span_m": (("blade_span_m",), "m"),
    "mass_kg": (("blade_mass_kg",), "kg"),
}


def characteristic_acceleration_g(density_kg_m3, velocity_m_s, chord_avg_m, span_m, mass_kg):
    """a_c = 0.5 * rho * U^2 * c_avg * b / m, returned in units of g."""
    a_c = 0.5 * density_kg_m3 * velocity_m_s ** 2 * chord_avg_m * span_m / mass_kg
    return a_c / G


def _lookup(meta, path):
    node = meta
    for key in path:
        if not isinstance(node, dict) or node.get(key) is None:
            return None
        node = node[key]
    return node


def resolve_inputs(meta, overrides=None):
    """Collects the five inputs for one run: an override (CLI) wins, then
    run_metadata.json. Returns (values, sources, missing) where values and
    sources are dicts keyed like INPUTS (sources: "cli" / "metadata") and
    missing lists the names that could not be found or are not usable
    (None, NaN, or not > 0)."""
    overrides = overrides or {}
    values, sources, missing = {}, {}, []
    for name, (path, _unit) in INPUTS.items():
        if overrides.get(name) is not None:
            v, src = overrides[name], "cli"
        else:
            v, src = _lookup(meta, path), "metadata"
        try:
            ok = v is not None and math.isfinite(float(v)) and float(v) > 0
        except (TypeError, ValueError):
            ok = False
        if ok:
            values[name], sources[name] = float(v), src
        else:
            missing.append(name)
    return values, sources, missing


def characteristic_acceleration_for_run(meta, overrides=None):
    """Returns (a_c_in_g_or_None, values, sources, missing)."""
    values, sources, missing = resolve_inputs(meta, overrides)
    if missing:
        return None, values, sources, missing
    return characteristic_acceleration_g(**values), values, sources, missing
