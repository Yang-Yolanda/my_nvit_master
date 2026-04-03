#!/home/yangz/.conda/envs/4D-humans/bin/python
import time
import os
import subprocess
import re
import argparse
from datetime import datetime

def check_process(pid):
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def check_log_error(log_path):
    """Scan log file for common errors."""
    if not os.path.exists(log_path):
        return "Log File Missing"
    
    with open(log_path, 'r') as f:
        # Read last 50 lines
        lines = f.readlines()[-50:]
        content = "".join(lines)
        
        if "Traceback" in content:
            return "Traceback detected"
        if "Error" in content or "error" in content:
            # Filter somewhat?
            if "Calculate accuracy..." in content: return None # Common log
            return "Error detected in logs"
        if "NaN" in content or "Grad norm nan" in content:
            return "NaN Gradients/Loss detected"
        if "Grad norm inf" in content:
            return "Inf Gradients detected"
            
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pids', nargs='+', type=int, help='List of PIDs to monitor')
    parser.add_argument('--logs', nargs='+', type=str, help='List of Log Paths')
    parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds')
    args = parser.parse_args()
    
    print(f"🕵️ Mr. A's Monitor Started at {datetime.now()}")
    print(f"   Monitoring PIDs: {args.pids}")
    print(f"   Monitoring Logs: {args.logs}")
    
    while True:
        status_report = []
        all_dead = True
        
        for pid, log_path in zip(args.pids, args.logs):
            is_running = check_process(pid)
            error = check_log_error(log_path)
            
            status = "🟢 Running" if is_running else "🔴 Dead"
            if error:
                status += f" ({error})"
            
            if is_running:
                all_dead = False
                
            status_report.append(f"PID {pid}: {status}")
            
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Report:")
        for line in status_report:
            print("   " + line)
            
        if all_dead:
            print("⚠️ All monitored processes are dead. Exiting Monitor.")
            break
            
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
