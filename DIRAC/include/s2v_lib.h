#ifndef S2V_LIB_H
#define S2V_LIB_H

#include "config.h"

int Init(const int argc, const char **argv);

void *GetGraphStruct();

int PrepareBatchGraph(void *_batch_graph,
                                 const int num_graphs,
                                 const int *num_nodes,
                                 const int *num_edges,
                                 void **list_of_edge_pairs,
                                 int is_directed);

int PrepareMeanField(void *_batch_graph,
                                void **list_of_idxes,
                                void **list_of_vals);

int PrepareLoopyBP(void *_batch_graph,
                              void **list_of_idxes,
                              void **list_of_vals);

int NumEdgePairs(void *_graph);

#endif