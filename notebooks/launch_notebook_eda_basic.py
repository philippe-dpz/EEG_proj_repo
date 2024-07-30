import os
import subprocess
import json
import sys

def update_json_and_open_notebook(filename):
    temp_filename = 'temp_filename.json'
    
    # Write the filename to the temporary JSON file
    with open(temp_filename, 'w') as f:
        json.dump({'filename': filename}, f)

    # Path to the Jupyter Notebook
    notebook_path = r'C:\Users\phili\EEG_project\EEG_proj_data\Exploratory\eda.ipynb'
    
    # Command to open the notebook in VS Code
    command = f'code "{notebook_path}"'
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python launch_notebook_eda_basic.py <filename>")
    else:
        filename = sys.argv[1]
        update_json_and_open_notebook(filename)