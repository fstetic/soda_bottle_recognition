import sys
import os
import pandas as pd
from pathlib import Path
import numpy as np
from hashlib import md5


def check_csv_correctness(df, data_dir):
	"""
	Checks if labels are correct and existence of that file
	:returns: None
	"""
	# for each row, i.e. pair label-file_path
	for index, row in df.iterrows():
		# file_path is a list ['.', 'directory', 'file_name']
		label, file_path = row['Label'], row['Path'].split('/')
		# check if the label is correct (label == directory name)
		if label != file_path[1]:
			print("File", file_path[2], "at index", index, "has a wrong label or is in the wrong directory!", file=sys.stderr)
		# also check if there exists a file with that path
		if not Path(data_dir + file_path[1] + '/' + file_path[2]).is_file():
			print("File", file_path[2], "at index", index, "does not exist!", file=sys.stderr)
	print("--Finished csv checkup--")


def check_duplicates(data_dir):
	"""
	Checks for duplicate files
	:return: None
	"""
	# dict hash : file_path
	unique = {}
	# traverse all directories in main directory
	for directory in ['M.Beer', 'MD.Diet', 'MD.Orig', 'P.Cherry', 'P.diet', 'P.Orig', 'P.Rsugar', 'P.Zero']:
		# traverse all files
		path = data_dir + directory
		for file_name in os.listdir(path):
			file_path = path + '/' + file_name
			with open(file_path, 'rb') as f:
				# hashing is done at once because files aren't large
				file_hash = md5(f.read()).hexdigest()
				if file_hash in unique.keys():
					print("Duplicates", (directory + '/' + file_name), "and", unique[file_hash], file=sys.stderr)
				else:
					unique[file_hash] = (directory + '/' + file_name)
	print("--Finished looking for duplicates--")


def check_label_consistency(df):
	"""
	Checks if there is really 8 labels and are they all spelled same
	:return: None
	"""
	# get label names
	labels, counts = np.unique(df['Label'], return_counts=True)
	if len(labels) != 8:
		# labels with smaller than average counts are wrong
		avg = np.average(counts)
		wrong = labels[counts < avg]
		print("There exist wrong labels:", wrong, file=sys.stderr)
	print("--Finished checking labels--")


# run file with dataset directory as argument to perform data cleaning
if __name__ == '__main__':
	# get dataset directory name
	if len(sys.argv) != 2:
		print("Please provide directory name.", file=sys.stderr)
		exit(1)
	directory_name = sys.argv[1]
	# read csv which has columns: Label,Path
	csv_file = pd.read_csv(directory_name + 'full.csv')
	# get directory path
	dataset_directory = os.getcwd() + '/' + directory_name

	check_label_consistency(csv_file)
	check_csv_correctness(csv_file, dataset_directory)
	check_duplicates(dataset_directory)