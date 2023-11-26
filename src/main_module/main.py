from utility_module.video_helpers import extract_frames
from utility_module.file_helpers import get_as_list_of_paths
from utility_module.file_helpers import find_next_directory
from utility_module.file_helpers import load_config
from ultralytics import YOLO
from torch import stack
from pathlib import Path
import argparse
import os

# constants
BAD_HAND = 1


# configs
root_dir = Path.cwd()
config = load_config(os.path.join(root_dir, "config.json"))

path_to_glitch_detector = config["path_to_glitch_detector"]
path_to_bad_guy_detector = config["path_to_bad_guy_detector"]
glitch_detector_conf_rate = config["glitch_detector_conf_rate"]
bad_guy_conf_values_mean_threshold = config["bad_guy_conf_values_mean_threshold"]
bad_hand_conf_value_threshold = config["bad_hand_conf_value_threshold"]
bad_hand_num_of_sequential_occurrence_threshold = config[
    "bad_hand_num_of_sequential_occurrence_threshold"
]
num_frames = config["num_frames"]


# initialize models
glitch_detector = YOLO(path_to_glitch_detector)
police = YOLO(path_to_bad_guy_detector)


def check_video(video: str, save_mode: bool = False):
    """
    :param video: path to video to be checked its authenticity
    :param save_mode: if true, save the prediction results
    :return: true if the function predicts the video to be true, false otherwise
    post-processing #1: decide that the video is fake depending on mean confidence results from classification
    post-processing #2: decide that the video is fake if a set number of detections are made for glitch
    """
    video_name = os.path.basename(video).strip(".mp4")
    print(f"Started deepfake prediction for {video_name}")

    # extract frames
    frames = extract_frames(video, num_frames=num_frames)
    print("Extracted frames from the video")

    # prediction
    results_police = police.predict(
        source=frames,
        verbose=False,
        save=save_mode,
        name=find_next_directory(f"{video_name}_", is_detect=False),
    )
    print("Analysed the video with the classification model")

    results_glitch = glitch_detector.predict(
        source=frames,
        conf=glitch_detector_conf_rate,
        verbose=False,
        save=save_mode,
        name=find_next_directory(f"{video_name}_", is_detect=True),
    )
    print("Analysed the video with the detection model")

    # post-processing #1
    bad_guy_conf_values_mean = (
        stack([result.probs.data[0] for result in results_police]).mean().item()
    )
    if bad_guy_conf_values_mean > bad_guy_conf_values_mean_threshold:
        print("Video: {video_name} Result: Fake")
        return False

    # post-processing #2
    bad_hand_count = 0
    for result in results_glitch:
        label_conf_dict = {
            label: conf
            for label, conf in zip(
                result.boxes.cls.tolist(), result.boxes.conf.tolist()
            )
        }

        if label_conf_dict.get(BAD_HAND, 0) > bad_hand_conf_value_threshold:
            bad_hand_count += 1
        else:
            bad_hand_count = 0

        if bad_hand_count > bad_hand_num_of_sequential_occurrence_threshold:
            print(f"Video: {video_name} Result: Real")
            return False

    print(f"Video: {video_name} Result: Real")
    return True


def run():
    parser = argparse.ArgumentParser(description="Deepfake detection script")
    parser.add_argument(
        "-f", "--file", type=str, help="Path to video file or directory", required=True
    )
    parser.add_argument(
        "--save",
        type=bool,
        help="1 for save prediction images, 0 otherwise",
        choices=[0, 1],
    )
    args = parser.parse_args()
    list_of_paths = get_as_list_of_paths(args.file)

    for i, path in enumerate(list_of_paths):
        check_video(path, args.save)


if __name__ == "__main__":
    run()
