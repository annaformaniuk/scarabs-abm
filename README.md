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



License
----

MIT
