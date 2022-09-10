#ifndef MVC_LIB_H
#define MVC_LIB_H

class sparseMatrix
{
 public:
    std::vector<int> rowIndex;
    std::vector<int> colIndex;
    std::vector<double> value;
    int rowNum;
    int colNum;
};

class mvc_lib{
public:
    void mvc_lib();
    void SetupGraphInput(std::vector<int> idxes, 
                         std::vector< std::shared_ptr<Graph> > g_list, 
                         std::vector< std::vector<int> > covered, 
                         const int* actions);
    int GetStatusInfo(std::shared_ptr<Graph> g, int num, const int* covered, int& counter, std::vector<int>& idx_map);
    sparseMatrix act_select;
    sparseMatrix rep_global;
    sparseMatrix n2nsum_param;
    sparseMatrix subgsum_param;
    std::vector< std::vector<int> > idx_map_list;
    std::vector< std::vector<double> > aux_feat;
    GraphStruct graph;
    std::vector<int> avail_act_cnt;
}

#endif