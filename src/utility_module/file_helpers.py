import os
import json
from utility_module import video_helpers


def load_config(file: str):
    """
    :return: configuration read from json `file`
    """
    with open(file, "r") as config_file:
        config = json.load(config_file)
    return config


def get_as_list_of_paths(path: str):
    """
    :return: list of file paths in `path` dir if path exists, otherwise `path` itself in list
    """
    if not os.path.exists(path):
        raise ValueError(f"{path} does not exist!")

    file_paths = []

    if os.path.isfile(path):
        file_paths.append(path)
    if os.path.isdir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            file_paths.append(file_path)

    return file_paths


def find_next_directory(base_name: str, is_detect: bool):
    """
    :param base_name: video file name
    :param is_detect: If true, directory will be under classify results. Otherwise, detect results
    :return: directory name with the highest order e.g. `base_name` 3 if `base_name` 2 exists
    this utils function is used to find a proper directory name for the results of the model predictions
    """
    parent_directory_path = "runs/detect" if is_detect else "runs/classify"

    # get existing directories with names matching the pattern
    existing_directories = [
        d
        for d in os.listdir(parent_directory_path)
        if os.path.isdir(os.path.join(parent_directory_path, d))
        and d.startswith(base_name)
    ]

    # find the highest order of the relevant directories
    existing_numbers = [
        int(d[len(base_name) :])
        for d in existing_directories
        if d[len(base_name) :].isdigit()
    ]
    highest_number = max(existing_numbers, default=0)

    return f"{base_name}{highest_number + 1}"


def process_directory(input_dir: str, output_dir: str, num_frames: int):
    """
    :param input_dir: path to directory containing mp4 video files
    :param output_dir: path to directory where extracted frames will be saved
    :param num_frames: number of frames to extract from each video
    this function recursively traverses all files in `input dir`, extracts frames from
    all video files, and saves the extracted frames in a directory structure that mirrors
    `input_dir` under `output_dir`
    """
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mp4"):
                # build the full paths for input and output
                input_file = os.path.join(root, file)
                output_sub_path = os.path.relpath(input_file, input_dir)
                output_path = os.path.dirname(os.path.join(output_dir, output_sub_path))

                # create the output directory if it doesn't exist
                os.makedirs(output_path, exist_ok=True)

                # extract frames and save the frames into output_path
                video_helpers.extract_frames(
                    input_file, output_path, mode=1, num_frames=num_frames
                )
