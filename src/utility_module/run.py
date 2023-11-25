import argparse
import os
from main_module import prodv1


parser = argparse.ArgumentParser(description="Deepfake detection script")
parser.add_argument("-f", "-file", help="Path to the video file(s)", required=True)
args = parser.parse_args()
if os.path.isfile(args.file):
    prodv1.detection(file_path=args.file)
elif os.path.isdir(args.file):
    for filename in os.listdir(args.file):
        file_path = os.path.join(args.file, filename)
        if os.path.isfile(file_path):
            prodv1(file_path)
else:
    print(f"The path {args.file} is not a valid file or directory.")

