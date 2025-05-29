# Functionality
Custom movement tracker that:
- takes a video of a behaviour arena,
- identifies the wells automatically,
- tracks the biggest object in each well
- computes the speed of the object in real time
- saves positions and speeds associated with time stamps in a .csv file for further processing in ActogramJ

## main version
This version has a GUI and can be used to fine tune the tracking parameters (threshold, grid size...).
It can also be used to track, coordinates and speeds are computed between each frame, but it will only be as fast as the real-time video for now.

## no-gui-version
This version is meant for computing and has the possibility to skip frames. Without skipping it should be about twice as fast as real-time. 

# Usage
Open the `tracker.py` script in a python notebook (vscode, jupyter lab...) give the correct path to the video to be tracked and start the scritp. If you are running the main version, a GUI will appear and it will be possible to set parameters in real time to check the effects on the video tracking. 
If you are running the no-gui-version the script will start immediately and estimate the duration of the processing in the terminal output. Therefore, I recommend adjusting the parameters with the GUI version first, copying them in to the no-gui-version and running the tracking with that. 
