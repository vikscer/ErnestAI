import os
import random


def equalize_directories(dir1, dir2):
    # Get list of files in each directory
    files1 = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
    files2 = [f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]

    # Get counts of files in each directory
    len1, len2 = len(files1), len(files2)

    # Determine which directory is larger and equalize file count
    if len1 > len2:
        files_to_remove = len1 - len2
        files_to_delete = random.sample(files1, files_to_remove)
        for file_name in files_to_delete:
            os.remove(os.path.join(dir1, file_name))
        print(f"Deleted {files_to_remove} files from {dir1} to match the file count of {dir2}.")

    elif len2 > len1:
        files_to_remove = len2 - len1
        files_to_delete = random.sample(files2, files_to_remove)
        for file_name in files_to_delete:
            os.remove(os.path.join(dir2, file_name))
        print(f"Deleted {files_to_remove} files from {dir2} to match the file count of {dir1}.")

    else:
        print("Both directories already have the same number of files.")


# Example usage
dir1 = "../wake_word_data/wake_word"
dir2 = "../wake_word_data/background_noise"
equalize_directories(dir1, dir2)
