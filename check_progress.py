import os

path = "D:/Memes2023_splitted_clean"
corrupt_files_path = 'unprocessed_files.txt'

num_files = len(os.listdir(path))
num_corrupt_files = len(open(corrupt_files_path).readlines())


with open('progress.txt','w') as file:
    file.write(str(num_files+num_corrupt_files)+'\n')