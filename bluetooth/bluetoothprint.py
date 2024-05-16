import asyncio
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
                print(
                    f"  Characteristic {characteristic.uuid} - {characteristic.description}, Handle: {characteristic.handle}")
                if characteristic.descriptors:
                    for descriptor in characteristic.descriptors:
                        print(f"    Descriptor {descriptor.uuid} - Handle: {descriptor.handle}")


# Run the asyncio event loop
asyncio.run(connect_and_explore(uuid))

