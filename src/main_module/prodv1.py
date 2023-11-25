from utility_module.extract_frames import extract_frames
from utility_module.arg_parser_helper import arg_helper
from utility_module.prediction_folder_naming_helper import find_next_directory
from ultralytics import YOLO
from torch import stack
import argparse
import os

# config
path_to_glitch_detector = "src/resources/models/glitch_detector_model.pt"
path_to_bad_guy_detector = "src/resources/models/best.pt"
glitch_detector_conf_rate = 0.4
bad_guy_conf_values_mean_threshold = 0.75
num_frames = 60


# initialize
glitch_detector = YOLO(path_to_glitch_detector)
police = YOLO(path_to_bad_guy_detector)


def check_video(path_to_input_video, idx, save_mode=False):
    video_name = os.path.basename(path_to_input_video)
    print(f"Started deepfake prediction for {video_name}\n")

    # extract frames
    frames = extract_frames(path_to_input_video, num_frames=num_frames)

    # prediction
    results_police = police.predict(
        source=frames,
        verbose=False,
        save=save_mode,
        name=find_next_directory(f"{video_name}_"),
    )
    results_glitch = glitch_detector.predict(
        source=frames,
        conf=glitch_detector_conf_rate,
        verbose=False,
        save=save_mode,
        name=find_next_directory(f"{video_name}_"),
    )

    # post-processing #1
    print("Evaluating results")
    bad_guy_conf_values_mean = (
        stack([result.probs.data[0] for result in results_police]).mean().item()
    )
    if bad_guy_conf_values_mean < bad_guy_conf_values_mean_threshold:
        print(f"{video_name} -> REAL\n\n")
        return

    print(f"{video_name} -> FAKE\n\n")
    return

    # post-processing #2
    bad_hand_count = 0
    for result in results_glitch:
        label_conf_dict = {
            label: conf
            for label, conf in zip(
                result.boxes.cls.tolist(), result.boxes.conf.tolist()
            )
        }

        if label_conf_dict.get(1, 0) > 0.5:
            bad_hand_count += 1
        else:
            bad_hand_count = 0

        if bad_hand_count > 3:
            print(f"{video_name} -> FAKE - glitch\n\n")
            return

    print(f"{video_name} -> REAL - end\n\n")


def main():
    parser = argparse.ArgumentParser(description="Deepfake detection script")
    parser.add_argument(
        "-f", "--file", help="Path to video file or directory", required=True
    )
    parser.add_argument(
        "--save",
        help="1 for save prediction images, 0 otherwise",
        type=bool,
        choices=[0, 1],
    )
    args = parser.parse_args()
    list_of_paths = arg_helper(args.file)

    for i, path in enumerate(list_of_paths):
        check_video(path, i, args.save)


if __name__ == "__main__":
    main()
