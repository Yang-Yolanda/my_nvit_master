#!/home/yangz/.conda/envs/4D-humans/bin/python

import wexpect as pexpect
import sys
import time

def deploy_and_run(mfa_code):
    host = "yangz@10.16.116.59"
    port = "2222"
    password = "13579yzr"
    
    script_path = "run_paper1_reeval.py"
    with open(script_path, "r", encoding='utf-8') as f:
        script_content = f.read()

    print(f"Connecting to {host}...")
    child = pexpect.spawn(f'ssh -p {port} {host}', encoding='utf-8', timeout=60)
    
    # 1. Login
    idx = child.expect(['password:', 'yes/no'], timeout=10)
    if idx == 1:
        child.sendline('yes')
        child.expect('password:')
    child.sendline(password)
    
    child.expect(['OTP', 'MFA', 'Verification'], timeout=10)
    child.sendline(mfa_code)
    
    print("MFA sent. Navigating menu...")
    
    # 2. Menu Interaction
    # Wait for prompt, send 'p' to refresh, find '3'
    # Wait for ANY prompt
    try:
        child.expect(['Opt>', '\[Host\]>', 'Search:', 'ID>'], timeout=20)
        # Refresh
        child.sendline('p')
        time.sleep(0.5)
        # Select 3
        child.sendline('3')
    except:
        print("Menu prompt ambiguous, blindly selecting 3...")
        child.sendline('3')
        
    # 3. Wait for Shell
    print("Waiting for remote shell...")
    child.expect(['\(4D-humans\)', 'yangz@'], timeout=30)
    print(">>> SHELL ACQUIRED <<<")
    
    # 4. Deploy Script
    print(f"Deploying {script_path} ({len(script_content)} bytes)...")
    remote_file = "remote_reeval_paper1.py"
    
    # Simple chunked echo
    # Escape tricky chars
    child.sendline(f'rm {remote_file}')
    
    lines = script_content.splitlines()
    # Batch echo to avoid overwhelming buffer
    batch_size = 20
    for i in range(0, len(lines), batch_size):
        chunk = lines[i:i+batch_size]
        # Construct a massive echo command? Or heredoc?
        # Heredoc is safer for large blocks
        # cat >> file << 'EOF'
        child.sendline(f"cat >> {remote_file} << 'EOF_PYTHON'")
        pass
        for line in chunk:
            # We must escape ONLY the terminator if it appears, which it won't.
            # But line endings need care? No, pexpect sendline adds \n.
            child.sendline(line)
        child.sendline("EOF_PYTHON")
        # Wait for prompt to ensure chunk written
        child.expect(['\(4D-humans\)', 'yangz@'], timeout=10)
        print(f"Uploaded chunk {i//batch_size + 1}...")

    # 5. Run it
    print("Running script on server...")
    child.sendline(f'python {remote_file}')
    
    # 6. Stream Output
    while True:
        try:
            child.expect('\n', timeout=300)
            line = child.before.strip()
            print(f"[REMOTE] {line}")
            if "[CONCLUSION]" in line and "VALID" in line:
                break
            if "ModuleNotFoundError" in line: # Stop on error
                print("Remote Error Detected.")
                break
        except pexpect.EOF:
            break
        except pexpect.TIMEOUT:
            pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deploy_and_run.py <MFA_CODE>")
    else:
        deploy_and_run(sys.argv[1])
