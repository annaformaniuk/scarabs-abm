# Scarabs-ABM

This is a Master's thesis project, meant to derive animal tracks from available dung beetle footage and integrate these tracks into an agent-based model that intends to describe and predict behaviour of the beetles. The project consists of three parts: extraction of dung beetle trajectories from videos, an agent-based model of dung-beetle behaviour, and sensitivity analysis, calibration, and validation of the agent-based model using the dung beetle trajectories. All of these parts can be executed separately, given that there is appropriate data at hand.

Requirements
----
- [NetLogo](https://ccl.northwestern.edu/netlogo/)
- [python](https://www.python.org/)
- [numpy](https://numpy.org/)
- [opencv](https://opencv.org/)
- [imutils](https://pypi.org/project/imutils/)
- [pyNetLogo](https://pynetlogo.readthedocs.io/en/latest/install.html)
- [matplotlib](https://matplotlib.org/3.1.1/users/installing.html)
- [scipy](https://www.scipy.org/)
- [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- [pandas](https://pandas.pydata.org/)
- [darknet](https://github.com/AlexeyAB/darknet)
- [holistically-nested edge detection](https://github.com/s9xie/hed)

## Usage

### The Agent-Based Model

Simply open the model at abm\code in the NetLogo modelling envorinment and let it run.
![model screenshot](https://github.com/annaformaniuk/scarabs-abm/blob/develop/images/model_calibrated_default.JPG "Model screenshot")

### Trajectory extraction

To reconstruct a trajectory, the beetle in a video must be filmed from above while it is rolling a dung ball. Darknet framework must be installed and the YOLOv3 weights present. 

- `trajectory-extraction/video_to_displacement_vectors.py`
Creates a json from the displacement vector between beetle object detected in video frames. The json has a following structure:
```
{
    "properties": [
        {
            "filename": "Allogymnopleuri_#05_Rolling from dung pat_20161119_cut",
            "ball_pixelsize": 44,
            "ball_realsize": 3.0,
            "fps": 30
        }
    ],
    "points": [
        {
            "point_coords": [
                371,
                1467
            ],
            "frame_number": 93,
            "displacement_vector": [
                0,
                0
            ]
        },
        {
            "point_coords": [
                348,
                1431
            ],
            "frame_number": 124,
            "displacement_vector": [
                -23,
                -36
            ]
        },
        ...
        
    ]
}
```

- To receive landscape of the video ground terrain, run the `trajectory-extraction/video_to_trajectory.py`, which outputs stitched together frames using homography transformation, as well as an approximation of obstacles using the holistically-nested edge detector. Note that the detector weights are needed for this, which can be found in the repository of authors. Because of accumulation of structural error, the stitching process will likely break when the stitched frame becomes too small for successive keypoint matching.

- Statistics of the reconstructed trajectories, such as speeds, travelled distance, duration of journey, chosen heading and histogram of heading deviations can be tested with `trajectory-extraction/trajectories_to_statistics.py`.

- Precision and recall of the YOLOv3 object detector can also be tested with `object_detection/yolo_validation_stats.py`.

### Agent-Based Model Calibration

- To choose which parameters to calibrate, the `sensitivity_analysis/sensitivity_analysis_full.py` can be run, which uses multiplication factor for each parameter and then plots dependency between the model state variables and those parameters.
- For calibration `calibration/brute_force.py` saves model results for all possible combination of parameters in the specified range. The model output has a following structure:
```
{
    "mean_speeds": 2.4883331592738274,
    "std_speeds": 0.3580032173753254,
    "heading_deviations": [3.090909090909091, 44.81818181818182, 223.27272727272728, 232.27272727272728, 140.27272727272728, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "heading_deviations_norm": [0.4801581697500353, 6.962293461375511, 34.6843666148849, 36.08247422680412, 21.790707527185425, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "mean_dist": 664.805685018406,
    "std_dist": 365.4904794704251,
    "mean_time": 322.125,
    "std_time": 215.39001688797,
    "protonum-width-impact": 1.0,
    "patch-roughness-impact": 2.5,
    "ball-roughness-impact": 2.5,
    "distance-threshold-impact": 0.1,
    "seen-radius-impact": 1.5
}
```
- Next, using the validation dataset of trajectories. `calibration/find_smallest_error.py` finds the smallest possible normalized root square mean error beetween the real and modelled statistics. 
- `validation/validation_plots.py` validates the model output using the validation datasets. It also plots how much the root square mean error for each state variable has decreased
![rmse plot](https://github.com/annaformaniuk/scarabs-abm/blob/develop/images/rmse_before_after.png "RMSE Plot")

- and what the mean values and their standard deviations are for the models
![speed results](https://github.com/annaformaniuk/scarabs-abm/blob/develop/images/results_speed.png "Speed results")

- `validation/scenarios_stats.py` also similarly analyses scenarios of how beetles behave when there's more of them present at the dung pile at the same time, and when there are additional obstacles created in the NetLogo world.

License
----

MIT
