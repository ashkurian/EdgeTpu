import subprocess
import os

def run_inference():
    model_file = "EdgeTpu/Automated_tfliteFiles/conv18.tflite"
    command = ["python3", "trial.py", "--model", model_file]

    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")

if __name__ == "__main__":
    run_inference()
