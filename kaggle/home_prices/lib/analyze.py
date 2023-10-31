def analyze(data):
    # Display the first few rows of the data to understand its structure
    print(data.head())

    # Get info about data types and missing values
    print(data.info())

    # Summary statistics for numerical features
    print(data.describe())

    # For categorical features
    print(data.describe(include=['object']))