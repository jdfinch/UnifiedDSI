
import subprocess

def git_current_commit():
    try:
        # Run the 'git rev-parse HEAD' command to get the current commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT)
        return commit_hash.strip().decode('utf-8')  # Decode to string and strip any extra whitespace
    except subprocess.CalledProcessError as e:
        return None



if __name__ == '__main__':
    current_commit = git_current_commit()
    print("Current Git commit:", current_commit)