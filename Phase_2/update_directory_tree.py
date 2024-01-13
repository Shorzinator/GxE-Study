import fnmatch
import os


def save_directory_tree(parent_dirs, ignore_dirs, output_file):
    with open(output_file, 'w') as f:
        for parent_dir in parent_dirs:
            for dirpath, dirnames, filenames in os.walk(parent_dir):
                # Remove directories in ignore_dirs
                dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
                for filename in fnmatch.filter(filenames, '*'):
                    f.write(os.path.join(dirpath, filename) + '\n')


# Specify the parent directories
parent_dirs = ['Phase_2']

# Specify the directories to ignore
ignore_dirs = ['.idea', '__pycache__', '.ipynb_checkpoints']

# Specify the output file
output_file = 'directory_tree.txt'

save_directory_tree(parent_dirs, ignore_dirs, output_file)
