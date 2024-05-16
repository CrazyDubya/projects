import asyncio
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


#49535343-6daa-4d02-abf6-19569aca69fe
#49535343-8841-43f4-a8d4-ecbe34729bb3
#49535343-aca3-481c-91ec-d85e28a60318
#49535343-1e4d-4bd9-ba61-23c647249616
