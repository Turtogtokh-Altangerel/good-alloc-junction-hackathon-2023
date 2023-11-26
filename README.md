<!-- ABOUT THE PROJECT -->
## Real or Rendered: The Authenticity Challenge
Welcome to the project made by the team `good_alloc` for Junction X Budapest 2023 Hackathon challenge.
This project proposes a multy-layer solution towards the Authenticity Challenge. In short, during evaluation mode of our 
solution, an input video is analysed and labeled while going through the following layers:
1. transform a video into a set number of frames
2. classify each frame, based on a custom trained YOLO model.
3. detect custom objects, based on a custom trapined YOLO model
4. analyse the outputs of the two predictions and finalize a decision

<!-- GETTING STARTED -->
## Getting Started
To try our solution locally, please set up our project locally by following the steps below.
### Prerequisites
Make sure you have Python version `>=3.10,<3.12.0` installed on your machine.

Make sure you have Poetry installed on your machine as well. If you use pip, you can install by
```sh
pip install poetry
```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Turtogtokh-Altangerel/good-alloc-junction-hackathon-2023.git
   ```
2. Jump to the root directory of the project

3. Install dependencies
   ```sh
   poetry install
   ```
   
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Input
**Note**: It is recommended to use a video, although not required, that the person drags their hand across their face. 
The format is required to be **_.mp4_** file. 

1. Detection on a single video
   ```sh
   poetry run good-detect -f path_to_video.mp4 
   ```

2. Detection on multiple videos. For this case, please provide the directory
   ```sh
   poetry run good-detect -f path_to_dir_of_videos
   ```

3. Saving results. For this case, please provide the flag `--save`. For example:
   ```sh
   poetry run good-detect -f path_to_video.mp4 --save 1
   ```
4. `config.json` at the root directory holds configuration values. The default values
are recommended, but you may edit the file and experiment with it. The file as default looks as
```json
{
  "path_to_glitch_detector": "src/resources/models/glitch_detector_model.pt",
  "path_to_bad_guy_detector": "src/resources/models/best.pt",
  "glitch_detector_conf_rate": 0.4,
  "bad_guy_conf_values_mean_threshold": 0.75,
  "bad_hand_conf_value_threshold": 0.5,
  "bad_hand_num_of_sequential_occurrence_threshold": 3,
  "num_frames": 60
}
```
5. In case you may want to run the project without poetry and its virtual environment, 
you can run the script as follows, for example,
   ```sh
   python3 src/main_module/main.py -f path_to_video.mp4
   ```
   
### Output
Video file name and its prediction are logged.
   

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Poetry setup
- [x] Frames extractor
- [x] Generic Authenticity classifier
- [x] Glitch detector
- [x] Verification layers
- [ ] Sign language detector

You can open issues at [open issues](https://github.com/Turtogtokh-Altangerel/good-alloc-junction-hackathon-2023/issues).

<!-- CONTACT -->
## Authors
team code: HyCKjO_kc
team name: good_alloc

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[poetry-badge]: https://img.shields.io/badge/packaging-poetry-cyan.svg