#!/usr/bin/env python3
"""
Step 2 verification: Modbus RTU test for AF10R0 accelerometer board.

Reads status, starts sampling, verifies FIFO data flow and XYZ values
from 4x LIS2DS12 sensors via RS485 at 921600 baud.

Usage:
    python test_modbus.py /dev/ttyUSB0                  # run tests
    python test_modbus.py /dev/ttyUSB0 stream            # stream to auto-named CSV
    python test_modbus.py /dev/ttyUSB0 stream data.csv   # stream to named CSV

Requirements:
    pip install minimalmodbus
"""

import csv
import os
import struct
import sys
import time
from datetime import datetime
import minimalmodbus

# --- Register addresses (must match modbus.h) ---

# Input registers (FC04)
REG_STATUS    = 0x0000
REG_AVAIL     = 0x0001
REG_OVERRUN   = 0x0002
REG_ODR       = 0x0003
REG_COUNTER_L = 0x0004
REG_COUNTER_H = 0x0005
REG_S1_X      = 0x0010
REG_BULK      = 0x0100

# Holding registers (FC06)
REG_CMD  = 0x0000
REG_ADDR = 0x0001

# Commands
CMD_START     = 1
CMD_STOP      = 2
CMD_RESET_BUF = 3

NUM_SENSORS = 4
REGS_PER_SET = 12  # 4 sensors x 3 axes
BULK_MAX_SETS = 84  # must match firmware (modbus.c)

# LIS2DS12 2g: 0.061 mg/LSB
MG_PER_LSB = 0.061

# --- Raw Modbus CRC16 (for bulk reads beyond minimalmodbus 125-reg limit) ---

def _generate_crc16_table():
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        table.append(crc)
    return table

_CRC16_TABLE = _generate_crc16_table()


def _modbus_crc16(data):
    crc = 0xFFFF
    for byte in data:
        crc = (crc >> 8) ^ _CRC16_TABLE[(crc ^ byte) & 0xFF]
    return crc


_retry_count = 0  # global retry counter for diagnostics
_retry_short = 0  # timeout / no response
_retry_crc = 0    # CRC mismatch (stale or corrupt bytes)


def _drain(serial_port, timeout=0.1):
    """Read and discard bytes until line is quiet for timeout seconds."""
    old_timeout = serial_port.timeout
    serial_port.timeout = timeout
    while serial_port.read(512):
        pass
    serial_port.timeout = old_timeout


def bulk_read_raw(serial_port, slave_addr, start_reg, num_regs, retries=3):
    """FC04 read bypassing minimalmodbus 125-register limit.

    Two-phase read: wait up to 500 ms for firmware to START responding
    (first byte), then read remaining data with short timeout.
    Normal reads still complete in ~25 ms.
    """
    global _retry_count, _retry_short, _retry_crc
    req = struct.pack('>BBHH', slave_addr, 0x04, start_reg, num_regs)
    crc = _modbus_crc16(req)
    req += struct.pack('<H', crc)

    # Response: addr(1) + fc(1) + byte_count(1) + data(num_regs*2) + crc(2)
    resp_len = 3 + num_regs * 2 + 2

    for attempt in range(retries):
        try:
            if attempt > 0:
                _drain(serial_port)  # consume any late response from prev attempt
                _retry_count += 1

            serial_port.write(req)

            # Phase 1: wait for firmware to start responding (may be in
            # a long SPI read cycle — up to 500 ms)
            serial_port.timeout = 0.5
            first = serial_port.read(1)
            if not first:
                raise IOError(f"short:0/{resp_len}")

            # Phase 2: firmware is transmitting — read rest at wire speed
            serial_port.timeout = 0.05
            rest = serial_port.read(resp_len - 1)
            resp = first + rest

            if len(resp) != resp_len:
                raise IOError(f"short:{len(resp)}/{resp_len}")

            if resp[1] & 0x80:
                raise IOError(f"exception:{resp[2]}")

            calc_crc = _modbus_crc16(resp[:-2])
            recv_crc = struct.unpack('<H', resp[-2:])[0]
            if calc_crc != recv_crc:
                raise IOError("crc")

            return list(struct.unpack(f'>{num_regs}H', resp[3:-2]))
        except IOError as e:
            if attempt < retries - 1:
                err = str(e)
                if "short" in err:
                    _retry_short += 1
                elif "crc" in err:
                    _retry_crc += 1
            else:
                raise


def read_register_raw(serial_port, slave_addr, reg, retries=3):
    """FC04 single register read via raw serial."""
    return bulk_read_raw(serial_port, slave_addr, reg, 1, retries=retries)[0]


def to_signed16(val):
    """Convert uint16 to int16."""
    return val - 65536 if val >= 32768 else val


def connect(port, slave_addr=1):
    """Set up minimalmodbus instrument."""
    instr = minimalmodbus.Instrument(port, slave_addr)
    instr.serial.baudrate = 921600
    instr.serial.timeout = 1.0
    instr.serial.parity = minimalmodbus.serial.PARITY_NONE
    instr.serial.stopbits = 1
    instr.serial.bytesize = 8
    instr.mode = minimalmodbus.MODE_RTU
    instr.clear_buffers_before_each_transaction = True
    return instr


def read_register_retry(instr, reg, functioncode=4, retries=3):
    """Read single register with retry on comm errors."""
    for attempt in range(retries):
        try:
            return instr.read_register(reg, functioncode=functioncode)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(0.1)


def read_registers_retry(instr, reg, count, functioncode=4, retries=3):
    """Read multiple registers with retry on comm errors."""
    for attempt in range(retries):
        try:
            return instr.read_registers(reg, count, functioncode=functioncode)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(0.1)


def write_register_retry(instr, reg, value, functioncode=6, retries=3):
    """Write single register with retry on comm errors."""
    for attempt in range(retries):
        try:
            instr.write_register(reg, value, functioncode=functioncode)
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(0.1)


def test_status(instr):
    """Test 1: Read status register, verify sensor OK flags."""
    print("\n--- Test 1: Status ---")
    status = read_register_retry(instr, REG_STATUS, functioncode=4)
    sensor_bits = status & 0x0F
    sampling = (status >> 8) & 1

    ok_count = 0
    for i in range(NUM_SENSORS):
        ok = bool(sensor_bits & (1 << i))
        print(f"  LIS{i+1}: {'OK' if ok else 'FAIL'}")
        if ok:
            ok_count += 1

    print(f"  Sampling: {'active' if sampling else 'idle'}")

    odr = read_register_retry(instr, REG_ODR, functioncode=4)
    print(f"  ODR: {odr} Hz")

    if ok_count == 0:
        print("  FAIL: No sensors detected")
        return False

    print(f"  PASS: {ok_count}/{NUM_SENSORS} sensors OK")
    return True


def test_start_sampling(instr):
    """Test 2: Start sampling via FC06 command."""
    print("\n--- Test 2: Start sampling ---")

    # Reset buffer first
    write_register_retry(instr, REG_CMD, CMD_RESET_BUF, functioncode=6)
    time.sleep(0.1)

    # Start sampling
    write_register_retry(instr, REG_CMD, CMD_START, functioncode=6)
    time.sleep(0.1)

    # Verify sampling is active
    status = read_register_retry(instr, REG_STATUS, functioncode=4)
    if not (status & 0x0100):
        print("  FAIL: Sampling not active after start command")
        return False

    print("  PASS: Sampling started")
    return True


def test_data_flow(instr):
    """Test 3: Verify data accumulates in buffer."""
    print("\n--- Test 3: Data flow ---")

    # Wait for MCU to complete a few FIFO read cycles
    time.sleep(0.5)

    avail = read_register_retry(instr, REG_AVAIL, functioncode=4)
    print(f"  Available samples: {avail}")

    if avail == 0:
        print("  FAIL: No samples after 500 ms")
        return False

    overrun = read_register_retry(instr, REG_OVERRUN, functioncode=4)
    print(f"  Overrun count: {overrun}")

    counter_l = read_register_retry(instr, REG_COUNTER_L, functioncode=4)
    counter_h = read_register_retry(instr, REG_COUNTER_H, functioncode=4)
    total = (counter_h << 16) | counter_l
    print(f"  Total counter: {total}")

    if total < 100:
        print(f"  WARN: Counter low ({total}), expected >300 at 1600 Hz after 500 ms")

    print(f"  PASS: Data flowing ({avail} available, {total} total)")
    return True


def test_latest_sample(instr):
    """Test 4: Read latest sample registers, verify plausible XYZ."""
    print("\n--- Test 4: Latest sample ---")

    regs = read_registers_retry(instr, REG_S1_X, REGS_PER_SET, functioncode=4)
    print(f"  Raw registers: {regs}")

    all_zero = True
    for s in range(NUM_SENSORS):
        base = s * 3
        x_raw = to_signed16(regs[base + 0])
        y_raw = to_signed16(regs[base + 1])
        z_raw = to_signed16(regs[base + 2])

        x_mg = x_raw * MG_PER_LSB
        y_mg = y_raw * MG_PER_LSB
        z_mg = z_raw * MG_PER_LSB

        if x_raw != 0 or y_raw != 0 or z_raw != 0:
            all_zero = False

        print(f"  S{s+1}: X={x_mg:+7.1f} mg  Y={y_mg:+7.1f} mg  Z={z_mg:+7.1f} mg")

    if all_zero:
        print("  FAIL: All readings are zero")
        return False

    # Check Z-axis: expect ~880-1000 mg at rest (gravity)
    for s in range(NUM_SENSORS):
        z_raw = to_signed16(regs[s * 3 + 2])
        z_mg = abs(z_raw * MG_PER_LSB)
        if z_mg < 500 or z_mg > 1300:
            print(f"  WARN: S{s+1} Z={z_mg:.0f} mg outside expected range (500-1300)")

    print("  PASS: Plausible XYZ values")
    return True


def test_bulk_read(instr):
    """Test 5: Bulk read 10 sample-sets (120 registers) from 0x0100."""
    print("\n--- Test 5: Bulk read ---")

    # Wait for enough data
    time.sleep(0.2)

    avail_before = read_register_retry(instr, REG_AVAIL, functioncode=4)
    print(f"  Available before bulk read: {avail_before}")

    if avail_before < 10:
        print(f"  SKIP: Not enough data ({avail_before} < 10)")
        return True

    num_sets = 10
    num_regs = num_sets * REGS_PER_SET  # 120
    regs = read_registers_retry(instr, REG_BULK, num_regs, functioncode=4)

    avail_after = read_register_retry(instr, REG_AVAIL, functioncode=4)
    consumed = avail_before - avail_after
    print(f"  Available after bulk read: {avail_after} (consumed ~{consumed})")

    # Print first and last sample-set
    for set_idx in [0, num_sets - 1]:
        base = set_idx * REGS_PER_SET
        vals = []
        for s in range(NUM_SENSORS):
            x = to_signed16(regs[base + s * 3 + 0]) * MG_PER_LSB
            y = to_signed16(regs[base + s * 3 + 1]) * MG_PER_LSB
            z = to_signed16(regs[base + s * 3 + 2]) * MG_PER_LSB
            vals.append(f"S{s+1}({x:+.0f},{y:+.0f},{z:+.0f})")
        print(f"  Set[{set_idx}]: {' '.join(vals)}")

    # Verify read pointer advanced
    if consumed < 1:
        print("  FAIL: Read pointer did not advance")
        return False

    print(f"  PASS: Bulk read OK, {consumed} sets consumed")
    return True


def test_overrun(instr):
    """Test 6: Verify overrun counter stays at 0 during normal operation."""
    print("\n--- Test 6: Overrun check ---")

    overrun = read_register_retry(instr, REG_OVERRUN, functioncode=4)
    print(f"  Overrun count: {overrun}")

    if overrun > 0:
        print(f"  WARN: {overrun} overruns detected (MCU not reading FIFO fast enough)")
    else:
        print("  PASS: No overruns")

    return True


def test_stop_sampling(instr):
    """Test 7: Stop sampling."""
    print("\n--- Test 7: Stop sampling ---")

    write_register_retry(instr, REG_CMD, CMD_STOP, functioncode=6)
    time.sleep(0.1)

    status = read_register_retry(instr, REG_STATUS, functioncode=4)
    if status & 0x0100:
        print("  FAIL: Sampling still active after stop command")
        return False

    print("  PASS: Sampling stopped")
    return True


def cmd_test(instr):
    """Run all hardware verification tests."""
    tests = [
        test_status,
        test_start_sampling,
        test_data_flow,
        test_latest_sample,
        test_bulk_read,
        test_overrun,
        test_stop_sampling,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test(instr):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    # Clean up: make sure sampling is stopped
    try:
        write_register_retry(instr, REG_CMD, CMD_STOP, functioncode=6)
    except Exception:
        pass

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


def set_low_latency(serial_port):
    """Reduce USB-serial adapter latency for streaming throughput.

    USB-serial adapters (FTDI, CH340, etc.) buffer received bytes and
    deliver them to the host in chunks.  The default latency timer is
    16 ms, which means a 245-byte Modbus response (split over ~4 USB
    fragments) takes 4 × 16 ms = 64 ms just in USB delivery latency.

    Method 1: ioctl ASYNC_LOW_LATENCY — works without root on FTDI.
    Method 2: sysfs latency_timer — requires write access (usually root).
    """
    # Method 1: ioctl ASYNC_LOW_LATENCY (no root needed on FTDI adapters)
    try:
        import array
        import fcntl

        TIOCGSERIAL = 0x541E
        TIOCSSERIAL = 0x541F
        ASYNC_LOW_LATENCY = 0x2000

        buf = array.array("i", [0] * 32)
        fcntl.ioctl(serial_port.fileno(), TIOCGSERIAL, buf)
        if not (buf[4] & ASYNC_LOW_LATENCY):
            buf[4] |= ASYNC_LOW_LATENCY
            fcntl.ioctl(serial_port.fileno(), TIOCSSERIAL, buf)
            print("  USB latency: set ASYNC_LOW_LATENCY")
        return True
    except (IOError, OSError, ImportError):
        pass

    # Method 2: sysfs latency_timer (needs root)
    dev = os.path.basename(serial_port.port)
    path = f"/sys/bus/usb-serial/devices/{dev}/latency_timer"
    try:
        with open(path, "r") as f:
            old = f.read().strip()
        with open(path, "w") as f:
            f.write("1")
        print(f"  USB latency timer: {old} ms -> 1 ms")
        return True
    except (IOError, OSError):
        return False


def cmd_stream(instr, csv_path):
    """Stream accelerometer data to CSV until Ctrl+C."""
    # Read status first
    status = read_register_retry(instr, REG_STATUS, functioncode=4)
    sensor_bits = status & 0x0F
    ok_count = sum(1 for i in range(NUM_SENSORS) if sensor_bits & (1 << i))
    if ok_count == 0:
        print("ERROR: No sensors detected")
        return False

    odr = read_register_retry(instr, REG_ODR, functioncode=4)
    print(f"Sensors: {ok_count}/{NUM_SENSORS} OK, ODR: {odr} Hz")

    # Reduce USB-serial latency (default 16 ms cripples throughput)
    if not set_low_latency(instr.serial):
        dev = os.path.basename(instr.serial.port)
        print(f"  WARN: Could not reduce USB latency. For best throughput:")
        print(f"    echo 1 | sudo tee /sys/bus/usb-serial/devices/{dev}/latency_timer")

    # Reset buffer and start sampling
    write_register_retry(instr, REG_CMD, CMD_RESET_BUF, functioncode=6)
    time.sleep(0.05)
    write_register_retry(instr, REG_CMD, CMD_START, functioncode=6)
    time.sleep(0.1)

    # Verify sampling started
    status = read_register_retry(instr, REG_STATUS, functioncode=4)
    if not (status & 0x0100):
        print("ERROR: Sampling did not start")
        return False

    print(f"Streaming to {csv_path}  (Ctrl+C to stop)")

    # Use raw serial exclusively in the streaming loop (mixing
    # minimalmodbus and raw serial causes intermittent failures)
    ser = instr.serial
    addr = instr.address

    total_samples = 0
    total_overruns = 0
    overrun_check = 0
    t_start = time.monotonic()

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            f.write(f"# LIS2DS12 2g mode, {MG_PER_LSB} mg/LSB, raw int16 values\n")
            writer.writerow([
                "sample", "time",
                "s1_x", "s1_y", "s1_z",
                "s2_x", "s2_y", "s2_z",
                "s3_x", "s3_y", "s3_z",
                "s4_x", "s4_y", "s4_z",
            ])

            while True:
                avail = read_register_raw(ser, addr, REG_AVAIL)
                if avail == 0:
                    time.sleep(0.001)
                    continue

                # Dynamic batch size: read whatever is available, up to max
                sets_to_read = min(avail, BULK_MAX_SETS)
                num_regs = sets_to_read * REGS_PER_SET
                regs = bulk_read_raw(ser, addr, REG_BULK, num_regs)
                t_batch = time.monotonic() - t_start

                # Batch convert + write
                signed = [v - 65536 if v >= 32768 else v for v in regs]
                for s in range(sets_to_read):
                    base = s * REGS_PER_SET
                    writer.writerow(
                        [total_samples, f"{t_batch:.4f}"]
                        + signed[base:base + REGS_PER_SET]
                    )
                    total_samples += 1

                # Check overrun every 10th iteration
                overrun_check += 1
                if overrun_check >= 10:
                    overrun_check = 0
                    total_overruns = read_register_raw(ser, addr, REG_OVERRUN)

                elapsed = time.monotonic() - t_start
                rate = total_samples / elapsed if elapsed > 0 else 0
                print(
                    f"\r[{elapsed:.1f}s] {total_samples} samples | "
                    f"{rate:.0f}/s | {total_overruns} overruns | "
                    f"err:{_retry_short}short/{_retry_crc}crc | \u2192 {csv_path}  ",
                    end="", flush=True,
                )

    except KeyboardInterrupt:
        print()  # newline after \r line

    # Restore settings for cleanup commands (minimalmodbus)
    ser.timeout = 1.0
    instr.clear_buffers_before_each_transaction = True

    # Stop sampling
    try:
        write_register_retry(instr, REG_CMD, CMD_STOP, functioncode=6)
    except Exception:
        pass

    elapsed = time.monotonic() - t_start
    rate = total_samples / elapsed if elapsed > 0 else 0
    print(f"Stopped. {total_samples} samples in {elapsed:.1f}s ({rate:.0f}/s), "
          f"{total_overruns} overruns")
    print(f"Saved to {csv_path}")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <serial_port> [stream [file.csv]]")
        print(f"  {sys.argv[0]} /dev/ttyUSB0                # run tests")
        print(f"  {sys.argv[0]} /dev/ttyUSB0 stream          # stream to CSV")
        print(f"  {sys.argv[0]} /dev/ttyUSB0 stream data.csv # stream to named CSV")
        sys.exit(1)

    port = sys.argv[1]
    command = sys.argv[2] if len(sys.argv) > 2 else "test"

    print(f"Connecting to {port} at 921600 baud")
    instr = connect(port)

    if command == "stream":
        csv_path = (
            sys.argv[3]
            if len(sys.argv) > 3
            else f"accel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        ok = cmd_stream(instr, csv_path)
        sys.exit(0 if ok else 1)
    elif command == "test":
        ok = cmd_test(instr)
        sys.exit(0 if ok else 1)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
