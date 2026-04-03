#!/home/yangz/.conda/envs/4D-humans/bin/python

import wexpect as pexpect
import sys
import time
import os

def connect_and_fix():
    print("=========================================")
    print("   AUTO-CORRECT CONNECT (Fixing Socket)")
    print("=========================================")
    
    # 1. New Path (Verified by manual check)
    socket_path = r"C:\Users\17545\mysock"
    host = "yangz@10.16.116.59"
    port = "2222"
    password = "13579yzr"

    # Cleanup Old
    if os.path.exists(socket_path):
        try: os.remove(socket_path)
        except: pass

    print(f"Target Socket: {socket_path}")
    
    # 2. Launch Master Connection
    # -M: Master mode
    # -S: Control Path
    # -tt: Force TTY (important for wexpect)
    cmd = f'ssh -tt -M -S {socket_path} -p {port} {host}'
    
    print(f"Spawning: {cmd}")
    child = pexpect.spawn(cmd, encoding='utf-8', timeout=60)
    
    # 3. Handle Login
    idx = child.expect(['password:', 'yes/no'], timeout=10)
    if idx == 1:
        child.sendline('yes')
        child.expect('password:')
    child.sendline(password)
    
    print("Password sent. Waiting for MFA...")
    
    # 4. Wait for MFA Prompt
    child.expect(['OTP', 'MFA', 'Verification', 'Code'], timeout=10)
    
    # 5. REQUEST MFA FROM USER
    print(">>> NEED_MFA <<<") # Signal
    sys.stdout.flush()
    
    # Read MFA from Stdin (User provides this)
    mfa_code = sys.stdin.readline().strip()
    print(f"Sending MFA: {mfa_code}")
    child.sendline(mfa_code)
    
    # 6. Menu & Keepalive
    print("Navigating Menu...")
    try:
        child.expect(['Opt>', '\[Host\]>', 'Search:'], timeout=20)
        child.sendline('p')
        time.sleep(0.5)
        child.sendline('3') # Supermicro
    except:
        child.sendline('3')

    print("Waiting for final prompt...")
    child.expect(['\(4D-humans\)', 'yangz@'], timeout=30)
    print(">>> CONNECTED <<<")
    print(f"Socket Established at: {socket_path}")
    
    # 7. Detach/Keepalive Loop
    # We must keep this process running to maintain the master socket
    print("Entering Keepalive Mode. DO NOT CLOSE.")
    while True:
        try:
            time.sleep(60)
            child.sendline('') # Keepalive
            child.read_nonblocking(size=1000, timeout=1)
        except:
            if not child.isalive():
                print("Connection died.")
                break

if __name__ == "__main__":
    connect_and_fix()
