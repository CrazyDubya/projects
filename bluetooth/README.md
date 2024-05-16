# Bluetooth Interaction Scripts

## Overview

This project contains a collection of Python scripts designed for discovering, connecting to, and interacting with Bluetooth devices using the Bleak library.

## Table of Contents

- [bluetooth.py](#bluetoothpy)
- [bluetoothprint.py](#bluetoothprintpy)
- [buetoothprintfind.py](#buetoothprintfindpy)
- [printtobluetoothcharac.py](#printtobluetoothcharacpy)
- [Future Features](#future-features)

## bluetooth.py

This script discovers and prints the names and addresses of nearby Bluetooth devices using the Bleak library.

### Usage

\```bash
python bluetooth.py
\```

\```python
from bleak import BleakScanner

async def scan():
    devices = await BleakScanner.discover()
    for d in devices:
        print(d.address, d.name)

asyncio.run(scan())
\```

## bluetoothprint.py

This script scans for Bluetooth devices and connects to a specific device by its UUID, then prints the services and characteristics of the connected device.

### Usage

\```bash
python bluetoothprint.py
\```

\```python
from bleak import BleakScanner, BleakClient

uuid = "13DC5F2B-AD12-9B87-E914-1C7FF95A0BE8"

async def connect_and_explore(uuid):
    print("Scanning for devices...")
    devices = await BleakScanner.discover()

    target_device = None
    for device in devices:
        if uuid.lower() in [str(s).lower() for s in device.metadata.get("uuids", [])]:
            target_device = device
            print(f"Found target device: {device.name}, {device.address}")
            break

    if not target_device:
        print("Target device not found. Please make sure the device is on and in range.")
        return

    async with BleakClient(target_device.address) as client:
        print("Connected. Discovering services and characteristics...")
        services = await client.get_services()
        for service in services:
            print(f"Service {service.uuid} - {service.description}")
            for characteristic in service.characteristics:
                print(f"  Characteristic {characteristic.uuid} - {characteristic.description}, Handle: {characteristic.handle}")
                if characteristic.descriptors:
                    for descriptor in characteristic.descriptors:
                        print(f"    Descriptor {descriptor.uuid} - Handle: {descriptor.handle}")

# Run the asyncio event loop
asyncio.run(connect_and_explore(uuid))
\```

## buetoothprintfind.py

This script connects to a Bluetooth device by its address and prints its services and characteristics, including descriptors.

### Usage

\```bash
python buetoothprintfind.py
\```

\```python
from bleak import BleakScanner, BleakClient

async def connect_and_explore(device_address):
    async with BleakClient(device_address) as client:
        is_connected = await client.is_connected()
        if is_connected:
            print(f"Connected to the device: {device_address}")

            services = await client.get_services()
            print("Services and characteristics:")
            for service in services:
                print(f"Service {service.uuid}:")
                for char in service.characteristics:
                    print(f"  Characteristic {char.uuid} (Handle: {char.handle}):")
                    for descriptor in char.descriptors:
                        value = await client.read_gatt_descriptor(descriptor.handle)
                        print(f"    Descriptor {descriptor.uuid} (Handle: {descriptor.handle}): {bytes(value)}")
        else:
            print(f"Failed to connect to the device: {device_address}")

# Replace with your device's address from the scan results
device_address = "13DC5F2B-AD12-9B87-E914-1C7FF95A0BE8"
asyncio.run(connect_and_explore(device_address))
\```

## printtobluetoothcharac.py

This script connects to a Bluetooth printer and sends a text message to be printed.

### Usage

\```bash
python printtobluetoothcharac.py
\```

\```python
from bleak import BleakClient

device_address = "13DC5F2B-AD12-9B87-E914-1C7FF95A0BE8"
write_characteristic_uuid = "49535343-1e4d-4bd9-ba61-23c647249616"

async def write_to_printer(device_address, write_characteristic_uuid, text):
    async with BleakClient(device_address) as client:
        is_connected = await client.is_connected()
        if is_connected:
            print("Connected to the printer")
            await client.write_gatt_char(write_characteristic_uuid, bytearray(text, 'utf-8'))
            print(f"Sent text to the printer: {text}")
        else:
            print("Failed to connect to the printer")

text_to_print = "Hello, printer!"
asyncio.run(write_to_printer(device_address, write_characteristic_uuid, text_to_print))
\```

## Future Features

### Bluetooth Interaction Enhancements
- **Extended Bluetooth Device Interaction:** Add functionality to interact with more Bluetooth device characteristics.
- **Improved Error Handling:** Enhance error handling to manage connection issues and device unavailability.
- **GUI Interface:** Develop a graphical user interface for easier Bluetooth device management and interaction.

---

Feel free to contribute to this project by submitting issues or pull requests.
