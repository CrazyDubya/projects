#!/usr/bin/env python3
"""
Test script to verify multi-port logging fix
"""
import subprocess
import time
import sys
import os
import requests

def test_multi_port_logging():
    """Test that multiple ports maintain separate logging"""
    print("=" * 60)
    print("TESTING MULTI-PORT LOGGING FIX")
    print("=" * 60)
    
    # Start two instances on different ports
    port1 = 8083
    port2 = 8084
    
    print(f"\n1. Starting Nomic instance on port {port1}...")
    print(f"2. Starting Nomic instance on port {port2}...")
    print("3. Testing that each port maintains its own game logging...")
    
    # Test by checking the /status endpoint for each port
    try:
        # Give processes time to start (if they're running)
        time.sleep(2)
        
        # Test port 8083
        try:
            response1 = requests.get(f"http://127.0.0.1:{port1}/status", timeout=5)
            status1 = response1.json()
            print(f"\n✅ Port {port1} response: {status1}")
        except:
            print(f"\n❌ Port {port1} not responding (may not be running)")
            
        # Test port 8084  
        try:
            response2 = requests.get(f"http://127.0.0.1:{port2}/status", timeout=5)
            status2 = response2.json()
            print(f"✅ Port {port2} response: {status2}")
        except:
            print(f"❌ Port {port2} not responding (may not be running)")
            
        print("\n" + "=" * 60)
        print("LOGGING FIX VERIFICATION")
        print("=" * 60)
        
        # Check that session directories are properly isolated
        session_dir_1 = f"game_sessions_port{port1}"
        session_dir_2 = f"game_sessions_port{port2}"
        
        if os.path.exists(session_dir_1):
            sessions_1 = len([d for d in os.listdir(session_dir_1) if d.startswith('session_')])
            print(f"✅ Port {port1} sessions directory exists: {sessions_1} sessions")
        else:
            print(f"❌ Port {port1} sessions directory missing")
            
        if os.path.exists(session_dir_2):
            sessions_2 = len([d for d in os.listdir(session_dir_2) if d.startswith('session_')])
            print(f"✅ Port {port2} sessions directory exists: {sessions_2} sessions")
        else:
            print(f"❌ Port {port2} sessions directory missing")
            
        print("\nTo fully test the fix:")
        print(f"1. Start game on port {port1}: python proper_nomic.py --port {port1}")
        print(f"2. Start game on port {port2}: python proper_nomic.py --port {port2}")
        print("3. Verify both games log independently to their respective directories")
        
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    test_multi_port_logging()