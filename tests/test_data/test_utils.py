from pathlib import Path


def get_test_data_path(filename: str) -> Path:
    """Get the full path to a test data file."""
    return Path(__file__).parent / filename


# Common test data that might be used across multiple tests
TEST_FILES = {
    "simple_race": "sample_race.cpp",
    # Add more test files here as needed
}
