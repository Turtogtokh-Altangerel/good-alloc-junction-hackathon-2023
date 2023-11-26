import cv2
import os


def extract_frames(video: str, output: str = "", mode: int = 0, num_frames: int = 16):
    """
    :param video: path to input video
    :param output: path to dir for extracted frames to be stored
    :param mode: set 1 to save the extracted frames in `output`
    :param num_frames: total num of frames to extract
    :return: extracted frames
    this function extracts a set number of frames from the input video
    """
    cap = cv2.VideoCapture(video)
    num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_size = max(num_frames_total // num_frames, 1)

    frame_num = 0
    count = 0
    frames = []

    # loop through the video frames and append a frame every step_size
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # end of the video

        if frame_num % step_size == 0:
            frames.append(frame)
            if mode == 1:
                video_name = os.path.basename(video).strip(".mp4")
                frame_path = os.path.join(output, video_name + f"_frame_{count}.jpg")
                cv2.imwrite(frame_path, frame)
                count += 1

        frame_num += 1

    cap.release()

    message = ""
    if output == "":
        message = "Video path: " + video + f"\nExtracted {len(frames)} frames\n"
    else:
        message = (
            "Video path: "
            + video
            + "\nOutput path: "
            + output
            + f"\n Extracted {len(frames)} frames\n"
        )
    print(message)

    return frames
