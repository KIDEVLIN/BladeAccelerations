# check_velocity.py  (run from scripts/aquire_data)
import wind_tunnel_conditions as wtc

zero = wtc.measure_zero_offsets(30)          # fan OFF
input("Fan ON and at speed, then press Enter... ")
fs, tr, sp, tv = wtc.read_raw_voltages(num_samples=5000)   # one 5 s mean

c_raw = wtc.compute_conditions(fs, tr, sp, tv, length_scale_m=0.02)
c_zer = wtc.compute_conditions(fs, tr, sp, tv, length_scale_m=0.02,
                               freestream_zero_v=zero.freestream_dp_v)

print(f"freestream raw mean : {fs:+.5f} V")
print(f"zero offset         : {zero.freestream_dp_v:+.5f} V")
print(f"corrected           : {c_zer.freestream_dp_v:+.5f} V")
print(f"dP (no zero / zero) : {c_raw.freestream_dp_pa:.2f} / {c_zer.freestream_dp_pa:.2f} Pa")
print(f"T                   : {c_zer.temp_c:.2f} C")
print(f"P static (abs)      : {c_zer.static_pressure_pa:.1f} Pa")
print(f"density             : {c_zer.density_kg_m3:.4f} kg/m^3")
print(f"velocity (no zero)  : {c_raw.velocity_freestream_m_s:.3f} m/s")
print(f"velocity (zeroed)   : {c_zer.velocity_freestream_m_s:.3f} m/s")