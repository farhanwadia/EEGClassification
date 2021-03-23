# Final Prototype Implementation Files

 - All these files MUST be kept in the same folder
 - If your GPU is not compatible with CUDA, you will need to set use_cuda = False
 - Make sure you change the save file path and subject initials before running
 
For the `output_ bluetooth.py` script, you will need to first pair the Bluetooth module with your computer. The code for Bluetooth pairing is 1234, but if that doesn't work, try 0000. The COM port may be different on your device, so check and change that if needed (may need different syntax for COM port on Mac).

In order to operate the valve, the python script `output_bluetooth.py` must be left continuously running on your computer and paired with the Bluetooth module on the circuit. The computer sends the input to the Bluetooth, not the mobile app. The mobile app will update an online database which the Python code is monitoring for updates to start recording. You will first need to set up the stream with Mind Monitor as you have for all other scripts (see the `Data Collection Scripts` folder for further details).