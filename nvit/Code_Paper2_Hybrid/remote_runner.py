#!/home/yangz/.conda/envs/4D-humans/bin/python

import subprocess
import sys
import time

if __name__ == "__main__":
    # Remote command to execute
    remote_cmd = "ls -F /home/yangz/NViT-master/nvit/"
    
    ssh_cmd = ["ssh", "-tt", "-p", "2222", "yangz@10.16.116.59", remote_cmd]
    
    print(f"Running Remote Metric Analysis: {remote_cmd}")
    
    process = subprocess.Popen(
        ssh_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0
    )
    
     # Send password
    try:
        time.sleep(2)
        print("Sending password...")
        process.stdin.write("13579yzr\n")
        process.stdin.flush()
        
        time.sleep(1)
        print("Sending MFA...")
        process.stdin.write("510234\n")
        process.stdin.flush()
        
        # Wait longer for analysis?
        stdout, stderr = process.communicate(timeout=60)
        print("--- STDOUT ---")
        print(stdout)
        print("--- STDERR ---")
        print(stderr)
        
    except subprocess.TimeoutExpired:
        print("Timeout! Analysis took too long or hung. Dumping partial.")
        process.kill()
        out, err = process.communicate()
        print("--- PARTIAL STDOUT ---", out)
        print("--- PARTIAL STDERR ---", err)

    except Exception as e:
        print(f"Error: {e}")
        process.kill()


