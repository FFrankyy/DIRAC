#ifndef MSG_PASS_H
#define MSG_PASS_H

#include "graph_struct.h"
#include "config.h"

sparseMatrix n2n_construct(GraphStruct* graph);

sparseMatrix e2n_construct(GraphStruct* graph);

sparseMatrix n2e_construct(GraphStruct* graph);

sparseMatrix e2e_construct(GraphStruct* graph);

sparseMatrix subg_construct(GraphStruct* graph);

#endif