import asyncio
from bleak import BleakScanner

print('h')

async def discover_devices():
    """Discovers nearby BLE devices and prints their addresses and names."""
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    if devices:
        print("Found the following devices:")
        for device in devices:
            print(f"Address: {device.address}, Name: {device.name if device.name else 'N/A'}")
    else:
        print("No BLE devices found.")

if __name__ == "__main__":
    asyncio.run(discover_devices())