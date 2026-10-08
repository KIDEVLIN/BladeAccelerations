#!/usr/bin/env python3
"""
.. module:: flash_modbus
   :platform: Linux, Windows
   :synopsis: Flash firmware over RS485 Modbus to AF10R1 bootloader.

.. moduleauthor:: Paul Bengtsson <paul@inunit.se>
"""
import time
import serial

# Workaround test
_WRITE_DELAY_S = 0.015

_orig_write = serial.Serial.write


def _write_with_delay(self, data):
    result = _orig_write(self, data)
    time.sleep(_WRITE_DELAY_S)
    return result


serial.Serial.write = _write_with_delay

from _flash_impl import main
main()
