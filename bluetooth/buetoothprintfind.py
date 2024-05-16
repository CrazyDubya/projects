import asyncio
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
