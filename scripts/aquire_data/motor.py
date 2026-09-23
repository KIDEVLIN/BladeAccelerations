"""
motor.py

Object-oriented wrapper around the SCL/velocity-mode servo drive control
code. Wraps the serial connection, the background encoder-polling thread,
and all the motion helper functions into a single Motor class.

Usage in your main script:

    from motor import Motor

    motor = Motor(port="COM4", baud=115200)
    motor.start()                      # opens serial, starts encoder thread

    print(motor.position())            # current angle in degrees
    motor.static_pattern([0, 5, 10, 15, 20], pause_time=3)
    motor.stop()                       # closes serial, stops thread

Or, using it as a context manager (recommended -- guarantees cleanup even
if an exception is raised):

    with Motor(port="COM4", baud=115200) as motor:
        motor.zero_position()
        motor.move(10)
        print(motor.position())
"""

import nidaqmx
import serial
import threading
import time


class Motor:
    def __init__(
        self,
        port="COM4",
        baud=115200,
        counts_per_deg=694.44,
        log_file="encoder_log.txt",
        poll_interval=0.005,
        daq_line="Dev11/port1/line0",
        max_travel_deg=360.0,
    ):
        self.port = port
        self.baud = baud
        self.counts_per_deg = counts_per_deg
        self.poll_interval = poll_interval
        self.daq_line = daq_line
        self.max_travel_deg = max_travel_deg

        self._ser = None

        # serial_lock guards every write+read exchange with the drive --
        # whether it's the background poller sending "IP" or a motion
        # command like "VE1"/"DI.../FL". Wrapping every full
        # request/response in this lock prevents interleaved bytes on the
        # wire and misrouted replies between threads.
        self._serial_lock = threading.Lock()

        # encoder_lock just guards the cached "latest known" reading, so
        # any other thread/method can grab the current value instantly
        # without touching serial.
        self._encoder_lock = threading.Lock()
        self._latest_counts = None
        self._latest_time = None

        self._stop_polling = threading.Event()
        self._poll_thread = None

        self._log_path = log_file
        self._log_file = None
        self._start_time = None

    # ----------------------------------------------------------------
    # Context manager support
    # ----------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    # ----------------------------------------------------------------
    # Lifecycle: start() / stop()
    # ----------------------------------------------------------------

    def start(self):
        """Open the serial port, open the log file, and start the
        background encoder-polling thread."""
        self._ser = serial.Serial(
            self.port, self.baud, timeout=0.1, write_timeout=0.1
        )
        self._log_file = open(self._log_path, "w", buffering=1)
        self._log_file.write("time_sec, encoder_counts\n")
        self._start_time = time.perf_counter()

        self._stop_polling.clear()
        self._poll_thread = threading.Thread(
            target=self._encoder_poll_loop, daemon=True
        )
        self._poll_thread.start()
        print("Background encoder polling/logging started.")

        time.sleep(0.2)  # give the poller a moment to land its first reading

    def stop(self):
        """Stop the polling thread and close the serial port / log file."""
        self._stop_polling.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2)
        print("Background encoder polling/logging stopped.")

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

        if self._ser is not None:
            self._ser.close()
            self._ser = None

    # ----------------------------------------------------------------
    # Low-level serial helpers
    # ----------------------------------------------------------------

    def _send(self, cmd, delay=0.02):
        """Write a command to the drive. Always goes through _serial_lock
        so it can never land on the wire in the middle of a background
        IP poll."""
        with self._serial_lock:
            print(f">> {cmd}")
            self._ser.write((cmd + "\r").encode())
            if cmd != "IP" and delay > 0:
                time.sleep(delay)

    def _send_and_read(self, cmd, delay=0.02):
        """Like _send, but also reads and returns whatever the drive
        replies with (used for query commands like 'CM')."""
        with self._serial_lock:
            print(f">> {cmd}")
            self._ser.write((cmd + "\r").encode())
            time.sleep(delay)
            data = self._ser.read(self._ser.in_waiting)
        return data.decode(errors="ignore").strip() if data else ""

    def _read_encoder_counts_locked(self):
        """Does the actual IP write/read/parse. Caller must already hold
        _serial_lock -- this method does not lock on its own."""
        self._ser.write(b"IP\r")
        time.sleep(0.0012)
        data = self._ser.read(self._ser.in_waiting)
        if not data:
            return None
        try:
            resp = data.decode(errors="ignore").strip()
            if "=" not in resp:
                return None
            hex_val = resp.split("=")[1]
            counts = int(hex_val, 16)
            if counts >= 0x80000000:
                counts -= 0x100000000
            return counts
        except Exception:
            return None

    # ----------------------------------------------------------------
    # Background encoder polling / logging thread
    # ----------------------------------------------------------------

    def _encoder_poll_loop(self):
        while not self._stop_polling.is_set():
            with self._serial_lock:
                counts = self._read_encoder_counts_locked()
            if counts is not None:
                t = time.perf_counter() - self._start_time
                with self._encoder_lock:
                    self._latest_counts = counts
                    self._latest_time = t
                self._log_file.write(f"{t:.6f}, {counts}\n")
            time.sleep(self.poll_interval)

    def get_latest_encoder(self):
        """Returns (counts, t_seconds) from whatever the background
        thread most recently read. Never blocks on serial I/O, so it's
        safe to call as often as you like from anywhere."""
        with self._encoder_lock:
            return self._latest_counts, self._latest_time

    # ----------------------------------------------------------------
    # Unit conversion
    # ----------------------------------------------------------------

    def degrees_to_counts(self, deg):
        return int(round(deg * self.counts_per_deg))

    def counts_to_degrees(self, counts):
        return counts / self.counts_per_deg

    # ----------------------------------------------------------------
    # Public position query -- this is your motor.position()
    # ----------------------------------------------------------------

    def position(self):
        """Current encoder position, in degrees. Returns None if no
        reading has landed yet."""
        counts, _ = self.get_latest_encoder()
        if counts is None:
            return None
        return self.counts_to_degrees(counts)

    # ----------------------------------------------------------------
    # Mode switching
    # ----------------------------------------------------------------

    def enter_velocity_mode(self):
        print("\n>>> Switching to Velocity Mode (CM11)\n")
        self._send("MD")
        self._send("CM11")
        self._send("ME")

    def enter_scl_mode(self):
        print("\n>>> Switching back to SCL/Jog Mode (CM10)\n")
        self._send("MD")
        self._send("CM10")
        self._send("ME")

    def get_current_mode(self):
        """Queries CM and returns it as an int, or None on parse failure."""
        resp = self._send_and_read("CM")
        try:
            return int(resp.replace("CM=", "").strip())
        except Exception:
            return None

    # ----------------------------------------------------------------
    # Motion
    # ----------------------------------------------------------------

    def zero_position(self):
        print("Zeroing position...")
        self._send("EP0")
        self._send("SP0")
        counts, _ = self.get_latest_encoder()
        if counts is not None:
            print(f"Position zeroed. Encoder now reads {counts} counts.")
        else:
            print("Warning: no encoder reading available yet.")

    def move(self, angle_deg):
        """Relative move by angle_deg. Raises ValueError if the resulting
        absolute position would exceed +/- max_travel_deg."""
        current_counts, _ = self.get_latest_encoder()
        if current_counts is not None:
            projected_deg = self.counts_to_degrees(current_counts) + angle_deg
            if abs(projected_deg) > self.max_travel_deg:
                raise ValueError(
                    f"Move by {angle_deg:+.2f} deg would put the motor at "
                    f"{projected_deg:+.2f} deg, outside the +/-{self.max_travel_deg} "
                    f"deg travel limit"
                )

        counts = self.degrees_to_counts(angle_deg)
        self._send("VE1")
        self._send(f"DI{counts}")
        self._send("FL")

    def move_to_angle(self, target_deg, wait=True, tolerance_deg=0.1,
                       stable_samples=3, timeout_s=10.0, poll_interval=0.01):
        """Absolute move to target_deg, computed relative to the last
        known encoder reading. Raises ValueError if target_deg is outside
        +/- max_travel_deg.

        If wait=True (default), blocks until the encoder position settles
        within tolerance_deg of target_deg for `stable_samples` consecutive
        fresh readings, or raises TimeoutError after timeout_s. Returns the
        settled position in degrees (or None if wait=False)."""
        if abs(target_deg) > self.max_travel_deg:
            raise ValueError(
                f"Target angle {target_deg:+.2f} deg is outside the "
                f"+/-{self.max_travel_deg} deg travel limit"
            )

        current_counts, _ = self.get_latest_encoder()
        if current_counts is None:
            print("No encoder reading yet -- can't compute a relative move.")
            return None

        target_counts = self.degrees_to_counts(target_deg)
        delta_counts = target_counts - current_counts

        print(f"Current angle: {self.counts_to_degrees(current_counts):.2f} deg")
        print(f"Target angle:  {target_deg:.2f} deg")
        print(f"Delta counts:  {delta_counts}")

        self._send("VE1")
        self._send(f"DI{delta_counts}")
        self._send("FL")

        if wait:
            return self.wait_until_settled(
                target_deg, tolerance_deg=tolerance_deg,
                stable_samples=stable_samples, timeout_s=timeout_s,
                poll_interval=poll_interval,
            )
        return None

    def wait_until_settled(self, target_deg, tolerance_deg=0.1,
                            stable_samples=3, timeout_s=10.0, poll_interval=0.01):
        """Block until the background-thread encoder cache reports a
        position within tolerance_deg of target_deg for `stable_samples`
        consecutive *fresh* readings (fresh = the poll timestamp advanced
        since the last check, so we're not re-checking a stale value while
        the motor is still moving between polls). Raises TimeoutError if
        that doesn't happen within timeout_s. No extra serial traffic --
        this just reads the cache the background thread already keeps."""
        deadline = time.time() + timeout_s
        consecutive_ok = 0
        last_seen_t = None

        while time.time() < deadline:
            counts, t = self.get_latest_encoder()
            if counts is not None and t != last_seen_t:
                last_seen_t = t
                deg = self.counts_to_degrees(counts)
                if abs(deg - target_deg) <= tolerance_deg:
                    consecutive_ok += 1
                    if consecutive_ok >= stable_samples:
                        return deg
                else:
                    consecutive_ok = 0
            time.sleep(poll_interval)

        counts, _ = self.get_latest_encoder()
        last_deg = self.counts_to_degrees(counts) if counts is not None else None
        raise TimeoutError(
            f"Motor did not settle at {target_deg:.2f} deg within {timeout_s}s "
            f"(last reading: {last_deg})"
        )

    def home(self, wait=True, timeout_s=15.0):
        """Return to the encoder zero reference. Call this at the end of
        every run (and from a finally block) so the physical zero stays
        consistent across power cycles."""
        print("\nHoming motor back to 0 deg...")
        return self.move_to_angle(0.0, wait=wait, timeout_s=timeout_s)

    def final_position(self):
        """Prints and returns the current angle, or None if unavailable."""
        deg = self.position()
        if deg is None:
            print("No encoder reading available.")
        else:
            print(f"Angle of Attack: {deg:.2f} degrees")
        return deg

    # ----------------------------------------------------------------
    # Static pattern
    # ----------------------------------------------------------------

    def static_pattern(self, angles, pause_time):
        """The background thread is already logging every sample
        continuously, so this just moves, waits, and (for the printed
        summary) samples the shared cache during the hold."""
        results = []

        for angle in angles:
            print(f"\nMoving to {angle}°")
            self.move_to_angle(angle)

            time.sleep(1.0)  # let the move happen; background thread logs it

            samples = []
            hold_start = time.time()
            while time.time() - hold_start < pause_time:
                counts, _ = self.get_latest_encoder()
                if counts is not None:
                    samples.append(counts)
                time.sleep(0.01)

            avg_deg = (
                (sum(samples) / len(samples)) / self.counts_per_deg
                if samples
                else None
            )
            results.append((angle, avg_deg))

        print("\n--- Static Pattern Results ---")
        for angle, avg in results:
            if avg is None:
                print(f"Angle {angle}° → No encoder data")
            else:
                print(f"Angle {angle}° → Avg Encoder = {avg:.3f}°")

        return results

    # ----------------------------------------------------------------
    # Wave / oscillation profile (velocity mode, externally driven)
    # ----------------------------------------------------------------

    def fire_trigger_pulse(self):
        try:
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(self.daq_line)
                task.start()
                task.write(True)
                time.sleep(0.010)  # 10 ms HIGH
                task.write(False)
            print("Trigger pulse fired.")
        except nidaqmx.errors.DaqError as e:
            print(f"DAQmx Error: {e}")

    def run_wave_profile(self):
        """Enters Velocity mode, waits for the drive to confirm CM11,
        fires the DAQ trigger pulse to kick off the externally-driven
        wave profile, then blocks on user input until 'exit' is typed."""
        self.enter_velocity_mode()
        print("\nAnalog-Velocity Mode Active.")

        while True:
            mode = self.get_current_mode()
            if mode == 11:
                break
            time.sleep(0.02)

        print("Motor confirmed in Velocity/Analog mode.")

        time.sleep(0.05)
        self.fire_trigger_pulse()

        print("Type 'Exit' to leave Analog-Velocity Mode.\n")
        while True:
            user_input = input(">>> ").strip()
            if user_input.lower() == "exit":
                break
            print("Type 'Exit' to leave Analog-Velocity Mode.")

        self.enter_scl_mode()
