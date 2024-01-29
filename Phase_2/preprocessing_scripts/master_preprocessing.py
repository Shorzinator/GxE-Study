import os
import subprocess

venv_path = "C:\\Users\\shour\\OneDrive\\Desktop\\GxE_Analysis\\venv"


def run_preprocessing(target):
    scripts = ["preprocessing_oldData_without_genetics.py",
               "preprocessing_oldData.py",
               "preprocessing_newData_without_genetics.py",
               "preprocessing_newData.py"]

    python_exe = os.path.join(venv_path, "Scripts", "python.exe")

    for script in scripts:
        subprocess.call([python_exe, script, target])


if __name__ == "__main__":
    target_1 = "AntisocialTrajectory"
    target_2 = "SubstanceUseTrajectory"

    run_preprocessing(target_1)
