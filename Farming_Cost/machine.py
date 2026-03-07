import os

def show_all_python_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            file_path = os.path.join(directory, filename)

            print(f"\n{filename}")
            print("-" * len(filename))

            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    print(file.read())
            except Exception as e:
                print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    folder_path = "."  # current directory
    show_all_python_files(folder_path)
