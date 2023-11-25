from utility_module.extract_frames import extract_frames
from ultralytics import YOLO
from torch import stack


def detection(file_path):
    path_to_glitch_detector = "src/resources/models/glitch_detector_model.pt"
    path_to_bad_guy_detector = "src/resources/models/best.pt"
    path_to_input_video = file_path

    # CONFIG


    glitch_detector_conf_rate = 0.4
    bad_guy_conf_values_mean_threshold = 0.75

    # extract frames
    frames = extract_frames(path_to_input_video, num_frames=60)

    # initialize
    glitch_detector = YOLO(path_to_glitch_detector)
    police = YOLO(path_to_bad_guy_detector)

    # prediction
    print("Started deepfake prediction\n")
    results_police = police.predict(source=frames, verbose=False)
    results_glitch = glitch_detector.predict(
        source=frames, conf=glitch_detector_conf_rate, verbose=False
    )

    # post-processing #1
    print("Evaluating results")
    bad_guy_conf_values_mean = (
        stack([result.probs.data[0] for result in results_police]).mean().item()
    )
    if bad_guy_conf_values_mean > bad_guy_conf_values_mean_threshold:
        print("BAD GUY DETECTED!")
        quit()

    # post-processing #2
    bad_hand_count = 0

    for result in results_glitch:
        label_conf_dict = {
            label: conf
            for label, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())
        }

        if label_conf_dict.get(1, 0) > 0.5:
            bad_hand_count += 1
        else:
            bad_hand_count = 0

        if bad_hand_count > 3:
            print("Too many bad hand count!")
            quit()
