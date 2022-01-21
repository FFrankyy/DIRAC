import os
import random
import numpy as np

lattice_dim = 4
lattice_scale = 4

def topKTruth(file_path, file_list, topk):
	res_improve_list_temp = []
	for file in file_list[:topk]:
		file = '%s/%s'%(file_path, file)
		res_improve = []
		for line in open(file):
			data = line.strip().split(',')
			res_improve.append(float(data[2].split(':')[1]))
		res_improve_list_temp.append(res_improve)
	res_improve_list = []
	for i in range(50):
		res_improve_temp = []
		for res_improve_item in res_improve_list_temp:
			res_improve_temp.append(res_improve_item[i])
		max_id = np.argmax(res_improve_temp)
		res_improve_list.append(res_improve_temp[max_id])
	return res_improve_list

# read gurobi results
res_gurobi = []
num = 0
for line in open('../TestData/Gurobi_res/Lattice_%dD_num_%d_Gurobi.txt'%(lattice_dim, lattice_scale), 'r'):
	if num < 50:
		data = line.strip().split(',')[1]
		res_gurobi.append(float(data.split(':')[1]))
	num += 1

# random read output results, and compare with Gurobi results
file_path = '../TestData/DQN_res/temp_data/%dD/%d'%(lattice_dim, lattice_scale)
file_list = os.listdir(file_path)

randomCases = 10
for i in range(randomCases):
	fout = open('../TestData/DQN_res/temp_data/%dD/Gurobi_match_count/%d/Lattice_%dD_num_%d_matchCount_randomCase_%d.txt'%(lattice_dim, lattice_scale, lattice_dim, lattice_scale, i), 'w')
	random.shuffle(file_list)
	topk = 0
	match_count_prev = 0
	stopCondition = True
	while stopCondition and topk < len(file_list):
		topk += 1
		res_rl = topKTruth(file_path, file_list, topk)
		match_count = np.sum([1 for item1, item2 in zip(res_gurobi, res_rl) if (item1 == item2)])
		if match_count > match_count_prev:
			print ('random case:%d, numInits:%d, match_count:%d'%(i, topk, match_count))
			fout.write('numInits:%d, match_count:%d, match_ratio:%.4f\n'%(topk, match_count, match_count/50))
			fout.flush()
		match_count_prev = match_count
		if match_count == 50:
			stopCondition = False
	fout.close()

