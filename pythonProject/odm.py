import os
import subprocess


def run_odm_for_folders(base_path):
    for entry in os.listdir(base_path):
        input_folder = os.path.join(base_path, entry)

        if os.path.isdir(input_folder):
            command = [
                "docker", "run",
                "-ti", "--rm",  # docker params
                "-v", f"{base_path}:/datasets",  # mount volume
                "opendronemap/odm",
                "--project-path", "/datasets",
                "--orthophoto-resolution", "1",
                "--end-with", "odm_orthophoto",
                entry
            ]

            print(f"Running ODM for {entry}...")
            subprocess.run(command, check=True)


if __name__ == '__main__':
    run_odm_for_folders(r"c:/Users/User/PycharmProjects/orthomosaic/pythonProject/filtered_group_frames")
