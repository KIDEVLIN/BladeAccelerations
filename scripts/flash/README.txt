AF10R1 Firmware Update Tool
===========================

Version:  1.0
Python:   3.11
Date:     2025-06-13

Updates AF10R1 firmware over RS485/Modbus RTU (921600 baud).


Installation
------------

1. Install Python 3.11 (https://www.python.org/downloads/)

2. Create a virtual environment and install dependencies:

   Linux / macOS:

     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt

   Windows:

     python -m venv venv
     venv\Scripts\activate
     pip install -r requirements.txt


Usage
-----

   python3 flash_modbus.py <PORT> <FIRMWARE.bin> [options]

Examples:

   Linux:
     python3 flash_modbus.py /dev/ttyUSB0 af10_app_00260501_02010101.bin

   Windows:
     python flash_modbus.py COM3 af10_app_00260501_02010101.bin

The device must be connected via RS485 (921600 baud, 8N1).
The script automatically puts the device into bootloader mode.


Options
-------

   --addr N          Modbus slave address (default: 1)
   --baud N          Baud rate (default: 921600)
   --skip-enter-bl   Skip ENTER_BL command (device already in bootloader)
   --no-header       Skip header/CRC injection (image already prepared)
   --pcb 0xNNNNNNNN  PCB version (default: 0x00260601)
   --ver 0xNNNNNNNN  App version (default: 0x02010100)


Files
-----

   flash_modbus.py    Entry point script
   _flash_impl.pyc    Compiled application logic (Python 3.11)
   requirements.txt   Python dependencies
   README.txt         This file
