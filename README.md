# RobotDrawing
A repository that allows a Franka arm to draw in a simulated environment. 

## How to use:

## Simulated Drawing:
- Download requirements from requirements.txt
- Optionally generate SVGs using generateSVG.py
- - Note that you must include your API key in sample.env. Rename this to .env.
- Run final_draw.py

## Actual Drawing:
- Download Franky and other dependencies
- Note that Franky *must* be recompiled from scratch to support Cartesian Impedance
- Run datalogging_betterrecovery.py
