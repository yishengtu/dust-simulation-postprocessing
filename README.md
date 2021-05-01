# dust-simulation-postprocessing

-----V2,4-----
- Removed spatial interpolation. Now only use the resolution output by ATHENA
- Improved interpolation method and logic. Current default interpolation method is "nearest"
- Improved loading and processing hydro logic. Now the time during the run will be within the processed hydro time (no extrapolation needed)

----- Known problem -----
- Can't run to the last hydro output time because of IndexError during loading hydro frames
- Numerical noise due to machine precision in ATHENA output (the outputted time is not exactly multiple of the value in athinput.hotcore, but slightly deviates. Usually higher)
