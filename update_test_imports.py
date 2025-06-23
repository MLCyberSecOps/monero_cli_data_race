import fileinput
import os


def update_imports(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with fileinput.FileInput(filepath, inplace=True) as f:
                        for line in f:
                            print(
                                line.replace("threadguard_new", "threadguard_enhanced"),
                                end="",
                            )
                    print(f"Updated: {filepath}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")


if __name__ == "__main__":
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    update_imports(test_dir)
