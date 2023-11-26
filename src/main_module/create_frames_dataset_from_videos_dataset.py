from utility_module.file_helpers import process_directory


# This script is used to create a dataset at `output_directory` with a structure mirroring dataset at `input_directory`
# and populate the output dataset with extracted frames from video files of the input dataset

# config
input_directory = "dataset/videos"
output_directory = "dataset/frames"

num_frames = 120
process_directory(input_directory, output_directory, num_frames)
