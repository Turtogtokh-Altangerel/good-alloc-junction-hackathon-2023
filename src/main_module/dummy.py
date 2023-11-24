import pandas as pd
import torch


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


if __name__ == "__main__":
    print_torch()
    print_data()
