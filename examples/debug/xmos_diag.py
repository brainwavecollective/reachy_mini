#!/usr/bin/env python3
"""Enhanced diagnostic script to check more XVF3800 parameters"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from reachy_mini.media.audio_control_utils import init_respeaker_usb, PARAMETERS

def diagnose_startup():
    """Check device state immediately after power-up"""
    print("=" * 60)
    print("XVF3800 Enhanced Startup Diagnostic")
    print("=" * 60)
    
    print("\n1. Attempting to connect to device...")
    dev = init_respeaker_usb()
    if not dev:
        print("ERROR: Device not found!")
        return
    print("✓ Device found")
    
    # Check ALL readable parameters
    print("\n2. Reading ALL device parameters:")
    print("-" * 60)
    
    for param_name, param_data in sorted(PARAMETERS.items()):
        access = param_data[3]
        if access == "wo":
            continue  # Skip write-only
        
        try:
            value = dev.read(param_name)
            print(f"  {param_name:40s}: {value}")
        except Exception as e:
            print(f"  {param_name:40s}: ERROR - {str(e)[:50]}")
    
    print("-" * 60)
    
    # Check specific boot status in detail
    print("\n3. Boot Status Details:")
    try:
        boot_status = dev.read("BOOT_STATUS")
        print(f"  Raw value: {boot_status}")
        print(f"  Type: {type(boot_status)}")
        if isinstance(boot_status, str):
            print(f"  As hex: {boot_status.encode('utf-8').hex()}")
    except Exception as e:
        print(f"  Error reading: {e}")
    
    dev.close()
    print("=" * 60)

if __name__ == "__main__":
    diagnose_startup()
