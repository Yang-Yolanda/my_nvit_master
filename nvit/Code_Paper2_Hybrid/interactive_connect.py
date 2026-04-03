#!/home/yangz/.conda/envs/4D-humans/bin/python

import wexpect as pexpect
import sys
import time
import os

def interactive_login():
    host = "yangz@10.16.116.59"
    port = "2222"
    password = "13579yzr"
    
    # 1. Start SSH
    print(f"Spawn command: ssh -p {port} {host}")
    child = pexpect.spawn(f'ssh -p {port} {host}', encoding='utf-8', timeout=60)
    
    # 2. Password
    idx = child.expect(['password:', 'yes/no'], timeout=10)
    if idx == 1:
        child.sendline('yes')
        child.expect('password:')
    child.sendline(password)
    
    # 3. Wait for MFA Prompt
    # We look for the exact prompt provided by the user previously: "[OTP Code]:"
    print("Waiting for MFA prompt...")
    child.expect(['OTP Code', 'MFA', 'Verification'], timeout=10)
    
    # 4. Request Code from User (via Agent)
    print(">>> NEED_MFA <<<") # Signal to Agent
    sys.stdout.flush()
    
    # 5. Read Code from Stdin (Agent will send this)
    mfa_code = sys.stdin.readline().strip()
    print(f"Sending MFA: {mfa_code}")
    child.sendline(mfa_code)
    
    # 6. Menu Navigation
    print("Navigating Menu...")
    # Refresh logic
    idx = child.expect(['Opt>', '\[Host\]>', 'Search:', 'ID>'], timeout=20)
    child.sendline('p') # Refresh
    time.sleep(1)
    
    # Select 3
    child.sendline('3')
    
    # 7. Shell
    print("Waiting for Shell...")
    child.expect(['\(4D-humans\)', 'yangz@'], timeout=30)
    print(">>> CONNECTED <<<")
    
    # 8. Run Evaluation (In-Line)
    # Echo script to file
    print("Deploying test script...")
    child.sendline('cat > reval.py <<EOF')
    child.sendline('import os; print("Re-Eval Check: dssp_experiments exists?", os.path.exists("dssp_experiments"))')
    child.sendline('EOF')
    
    child.sendline('python reval.py')
    child.expect('dssp_experiments', timeout=10)
    print(child.before + child.after)
    
    # 9. Interactive Loop (allow Agent to send more commands)
    print("Entering command loop. Type 'exit' to stop.")
    while True:
        cmd = sys.stdin.readline().strip()
        if not cmd or cmd == 'exit': break
        child.sendline(cmd)
        # Read a bit
        try:
             # Basic read loop
             time.sleep(0.5)
             print(child.read_nonblocking(size=1000, timeout=1))
        except:
            pass

if __name__ == "__main__":
    interactive_login()
