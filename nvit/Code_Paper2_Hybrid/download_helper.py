#!/home/yangz/.conda/envs/4D-humans/bin/python
import wexpect
import sys
import os

def download_file(remote_path, local_path):
    print(f"Downloading {remote_path} to {local_path}...")
    # scp -P 2222 yangz@10.16.116.59:remote_path local_path
    cmd = f'scp -P 2222 yangz@10.16.116.59:{remote_path} {local_path}'
    child = wexpect.spawn(cmd)
    
    try:
        idx = child.expect(['password:', 'yes/no'], timeout=30)
        if idx == 1:
            child.sendline('yes')
            child.expect('password:')
        
        child.sendline('13579yzr')
        
        child.expect('MFA Code:', timeout=30)
        print("Please provide MFA Code:")
        # In this environment, we usually get MFA via user or system
        # Since I'm an agent, I'll wait for a moment or hope the system handles it?
        # Actually, let's just use the pexpect_ssh structure we have.
    except Exception as e:
        print(f"Error: {e}")
        child.terminate()

if __name__ == "__main__":
    # This is a template, real execution might need user MFA input
    pass
