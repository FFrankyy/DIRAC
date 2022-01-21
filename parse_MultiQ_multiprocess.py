import numpy as np
import os
from config import parsers

args = parsers()

def parse_result(lattice_dim, test_scale, stepRatio):
    res_list, res_improve_list, time_list, Qnum_list = [], [], [], []
    fout = open('../TestData/DQN_res/Lattice_%dD_num_%d_MultiQ.txt'%(lattice_dim, test_scale), 'w')
    for i in range(50):
        for line in open('../TestData/DQN_res/temp_data/Lattice_%dD_num_%d_stepRatio_%.4f_MultiQ_gid_%d.txt'%(lattice_dim, test_scale, stepRatio, i), 'r'):
            data = line.strip().split(',')
            res = float(data[1].strip().split(':')[1])
            res_improve = float(data[2].strip().split(':')[1])
            tim = float(data[3].strip().split(':')[1])
            Qnum = int(data[4].strip().split(':')[1])
            res_list.append(res)
            res_improve_list.append(res_improve)
            time_list.append(tim)
            Qnum_list.append(Qnum)
            fout.write('Graph: %d, result: %.4f, improved result: %4f, time: %.2f, Qnum: %d\n'%(i, res, res_improve, tim, Qnum))
            fout.flush()
    fout.write('Result: %.4f+%.4f, Time:%.2f+%.2f, Qnum:%.2f+%.2f\n'%(np.mean(res_list), np.std(res_list), np.mean(time_list), np.std(time_list), np.mean(Qnum_list), np.std(Qnum_list)))
    fout.write('Result_improve: %.4f+%.4f'%(np.mean(res_improve_list), np.std(res_improve_list)))
    fout.close()

if __name__=="__main__":

    parse_result(args.lattice_dim, args.test_scale, args.stepRatio)
