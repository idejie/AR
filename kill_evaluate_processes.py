import subprocess
import os
import time

def kill_processes(process_name):
    try:
        # Get all processes
        cmd = "ps -ef"
        output = subprocess.check_output(cmd, shell=True).decode().split('\n')

        pids_to_kill = []
        for line in output:
            if process_name in line and 'grep' not in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    pids_to_kill.append(pid)

        if not pids_to_kill:
            print(f"No processes found for '{process_name}'.")
            return

        print(f"Found processes to kill: {pids_to_kill}")
        for pid in pids_to_kill:
            try:
                os.kill(int(pid), 9)  # SIGKILL
                print(f"Killed process {pid}")
            except OSError as e:
                print(f"Error killing process {pid}: {e}")
        
        # Verify if processes are killed
        time.sleep(1) # Give some time for processes to terminate
        remaining_processes = []
        cmd = "ps -ef"
        output = subprocess.check_output(cmd, shell=True).decode().split('\n')
        for line in output:
            if process_name in line and 'grep' not in line:
                remaining_processes.append(line)
        
        if remaining_processes:
            print(f"Some processes for '{process_name}' are still running:")
            for line in remaining_processes:
                print(line)
        else:
            print(f"All processes for '{process_name}' have been killed.")

    except subprocess.CalledProcessError as e:
        print(f"Error running ps command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    kill_processes("evaluate.py") 