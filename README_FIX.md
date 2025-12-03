# Project Fixes

I have fixed the project by performing the following actions:

1.  **Restored `requirements.txt`**: The file was empty. I populated it with the dependencies from `requirements_training.txt`.
2.  **Installed Dependencies**: I ran `pip install -r requirements.txt` to ensure all necessary libraries are installed.
3.  **Verified API**: I ran the API server and performed a quick test (`quick_api_test.py`) to confirm it is working correctly.
4.  **Created Helper Scripts**:
    *   `setup_project.bat`: Run this to set up the virtual environment and install dependencies.
    *   `run_api.bat`: Run this to start the API server.

## How to Run

1.  Run `setup_project.bat` (if you haven't already).
2.  Run `run_api.bat` to start the backend server.
3.  Load the extension in Chrome:
    *   Go to `chrome://extensions/`
    *   Enable "Developer mode"
    *   Click "Load unpacked"
    *   Select the `app` folder in this project.

The API server must be running for the extension to work.
