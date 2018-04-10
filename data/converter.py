
# -*- coding:utf-8 -*-
if __name__ == "__main__":
	fp = './math_equation_data_no_spaces.txt'
	outfp = './math_equation_data_3.txt'
	newlines = []
	with open(fp, 'r') as f:
		newlines = [line[0:3] for line in f.readlines()]

	num_lines = len(newlines)
	with open(outfp, 'w+') as f:
		for i,line in enumerate(newlines):
			if i == num_lines-1:
				break
			f.write(line + "\n")



	


