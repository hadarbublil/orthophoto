import os
import subprocess


def run_odm_for_folders(base_path):
    for entry in os.listdir(base_path):
        input_folder = os.path.join(base_path, entry)

        if os.path.isdir(input_folder):
            output_folder = os.path.join(base_path, f"odm_output_{entry}")

            os.makedirs(output_folder, exist_ok=True)

            command = [
                "docker", "run", "--rm",
                "-v", f"{input_folder}:/mnt/input",
                "-v", f"{output_folder}:/mnt/output",
                "opendronemap/odm",
                "--project-path", "/mnt/input",
                "--orthophoto",
                "--copy-to", "/mnt/output"
            ]

            print(f"Running ODM for {entry}...")
            subprocess.run(command, check=True)
