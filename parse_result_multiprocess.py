import numpy as np
import os
from config import parsers

args = parsers()

def parse_result(file_path, model_name, lattice_dim, lattice_scale, cut_num):
    file_path = '../TestData/DQN_res/temp_data/%dD/%d'%(lattice_dim, lattice_scale)
    file_list = os.listdir(file_path)

    if cut_num > len(file_list):
        print ('invalid cut_num, set cut_num to be len(file_list)')
        cut_num = len(file_list)

    res_list_temp, res_improve_list_temp, time_list_temp = [], [], []
    for file in file_list[:cut_num]:
        file = '%s/%s'%(file_path, file)
        res, res_improve, time = [], [], []
        for line in open(file):
            data = line.strip().split(',')
            res.append(float(data[1].split(':')[1]))
            res_improve.append(float(data[2].split(':')[1]))
            time.append(float(data[3].split(':')[1]))
        res_list_temp.append(res)
        res_improve_list_temp.append(res_improve)
        time_list_temp.append(time)

    fout = open('../TestData/DQN_res/%s_num_%d_stepRatio_0.0100_MultiQ_%dInits.txt'%(model_name, lattice_scale, cut_num), 'w')
    res_list, res_improve_list, time_list = [], [], []
    for i in range(50):
        res_temp, res_improve_temp, time_temp = [], [], []
        for res_item in res_list_temp:
            res_temp.append(res_item[i])
        for res_improve_item in res_improve_list_temp:
            res_improve_temp.append(res_improve_item[i])
        for time_item in time_list_temp:
            time_temp.append(time_item[i])

        max_id = np.argmax(res_improve_temp)
        res_list.append(res_temp[max_id])
        res_improve_list.append(res_improve_temp[max_id])
        time_list.append(time_temp[max_id])

        fout.write('Graph: %d, result: %.4f, improved result: %.4f, time:%.2f\n' % (i, res_temp[max_id], res_improve_temp[max_id], time_temp[max_id]))
        fout.flush()
    fout.write('Result:%.4f+%.4f, Time:%.2f+%.2f\n' % (np.mean(res_list), np.std(res_list), np.mean(time_list), np.std(time_list)))
    fout.write('Result_improve:%.4f+%.4f\n' % (np.mean(res_improve_list), np.std(res_improve_list)))
    fout.close()

if __name__=="__main__":

    file_path = '../TestData/DQN_res/temp_data/%dD/%d'%(args.lattice_dim, args.test_scale)
    parse_result(file_path, args.model_name, args.lattice_dim, args.test_scale, args.cut_num)