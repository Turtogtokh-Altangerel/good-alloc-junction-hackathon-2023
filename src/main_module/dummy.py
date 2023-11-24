import pandas as pd


def main():
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


if __name__ == "__main__":
    main()
