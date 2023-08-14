".py file to run multiple inferences corresponding to each model .tflite file in the folder "EdgeTpu/Automated_tfliteFiles/"
Use "python3 run_inference_multiple.py" to run this .py file


import subprocess
import sys
import os
import time

def run_inference(model_file):
    command = ["python3", "trial.py", "--model", model_file]

    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"Error running inference for model {model_file}: {e}")

if __name__ == "__main__":
    model_files_dir = "EdgeTpu/Automated_tfliteFiles/"

    # Get a list of all TFLite model files in the directory
    model_files = [file for file in os.listdir(model_files_dir) if file.endswith(".tflite")]

    if not model_files:
        print("No TFLite model files found in the directory.")
        sys.exit(1)

    num_runs = 50  # Change this number to the desired number of runs

    for run in range(1, num_runs+1):
        for model_file in model_files:
            model_path = os.path.join(model_files_dir, model_file)
            print(f"Running inference for model {model_file}, run {run}/{num_runs}")
            run_inference(model_path)
            print(f"Finished inference for model {model_file}, run {run}/{num_runs}")

            # Add a delay of 5 seconds between inference runs (adjust as needed)
            time.sleep(5)
