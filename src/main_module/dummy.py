import pandas as pd
import torch
from ultralytics import YOLO


def print_data():
    # Create a DataFrame
    data = {
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "City": ["New York", "San Francisco", "Los Angeles"],
    }

    df = pd.DataFrame(data)

    # Display the DataFrame
    print("DataFrame:")
    print(df)


def print_torch():
    x = torch.rand(5, 3)
    print(x)


def sanity_yolo():
    model = YOLO("yolov8n.pt")
    results = model(
        "https://ultralytics.com/images/bus.jpg", save=True, save_crop=True
    )  # predict on an image


if __name__ == "__main__":
    # print_torch()
    # print_data()
    sanity_yolo()
