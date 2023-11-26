from utility_module.video_helpers import extract_frames


# This is a sample script to extract frames from a video and save the results

input_file = ""  # give path to input video
output_path = ""  # give path to dir where extracted frames will be stored
num_frames = 60

# extract frames and save the frames into output_path
extract_frames(input_file, output_path, mode=1, num_frames=num_frames)
