#include "PrepareBatchGraph.h"
#include <stdio.h>
sparseMatrix::sparseMatrix()
{
    rowNum = 0;
    colNum = 0;
    rowIndex.clear();
    colIndex.clear();
    value.clear();
}

 PrepareBatchGraph::PrepareBatchGraph(int _aggregatorID)
{
    aggregatorID = _aggregatorID;
}


void PrepareBatchGraph::SetupGraphInput(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list, 
                           std::vector< std::vector<int> > covered, 
                           const int* actions,
                           int PE_dim)
{
    act_select = std::shared_ptr<sparseMatrix>(new sparseMatrix());
    rep_global = std::shared_ptr<sparseMatrix>(new sparseMatrix());

    int node_cnt = 0, edge_cnt = 0;
//    std::vector<double> degrees;
    std::vector<double> node_states;

    for (size_t i = 0; i < idxes.size(); ++i)
    {
        auto g = g_list[idxes[i]];
        node_cnt += g->num_nodes;
        for(int j=0;j<g->num_nodes; ++j)
        {
//            degrees.push_back((double)g->adj_list[j].size() / (double)g->num_nodes);
//              node_states.push_back((double)g->adj_list[j].size() / (double)g->num_nodes);
            node_states.push_back((double)g->node_state[j]);
        }
        edge_cnt += g->num_edges * 2;
    }
    graph.Resize(idxes.size(), node_cnt);

    if (actions)
    {
        act_select->rowNum = idxes.size();
        act_select->colNum=(size_t)node_cnt;
    } else
    {
        rep_global->rowNum = (size_t)node_cnt;
        rep_global->colNum = idxes.size();
    }
    node_cnt = 0;
    edge_cnt = 0;
    size_t edge_offset = 0;
    for (size_t i = 0; i < idxes.size(); ++i)
	{
        auto g = g_list[idxes[i]];
        std::set<int> c;

        for (size_t j = 0; j < covered[idxes[i]].size(); ++j)
        {
            auto cc = (covered[idxes[i]]);
            int n_c = cc[j];
            c.insert(n_c);
        }

        for (int j = 0; j < g->num_nodes; ++j)
        {
            // define node coords feat
            std::vector<double> temp_node_feat;

            //for (size_t k = 0; k < g->node_coords_dim; ++k)
            //{
            //    int pos = g->node_coords[j*g->node_coords_dim+k];
            //    for(int e =0; e<PE_dim; ++e)
            //    {
            //      double p_embedding = 0.0;
            //       if(e%2==0){
            //          p_embedding = (double)sin(pos/pow(10000,2.0*e/PE_dim));
            //        }
            //       else
            //       {
            //           p_embedding = (double)cos(pos/pow(10000,2.0*e/PE_dim));
            //      }
            //        temp_node_feat.push_back(p_embedding);
            //    }
            //}

            for(size_t k = 0; k < g->node_coords_dim; ++k)
            {
                temp_node_feat.push_back((double)(g->node_coords[j*g->node_coords_dim+k]/g->num_nodes));
           ////     temp_node_feat.push_back(1.0);
            }

            node_feat.push_back(temp_node_feat);

            int x = node_cnt + j;
            graph.AddNode(i, x);

            // define edge feat
            for (auto p : g->adj_list[j])
            {
                std::vector<double> temp_edge_feat;
                graph.AddEdge(edge_cnt, x, node_cnt + p.first);

                temp_edge_feat.push_back(p.second);    // edge weight
                 // position encode edge weight
//                double pos = p.second;
//                for(int e =0; e<PE_dim; ++e)
//                {
//                   double p_embedding = 0.0;
//                   if(e%2==0){
//                       p_embedding = (double)sin(pos/pow(10000, 2.0*e/PE_dim));
//                   }
//                   else
//                   {
//                       p_embedding = (double)cos(pos/pow(10000, 2.0*e/PE_dim));
//                   }
//                    temp_edge_feat.push_back(p_embedding);
//                }

                temp_edge_feat.push_back(c.count(j));
                temp_edge_feat.push_back(c.count(p.first) ^ c.count(j));
//                temp_feat.push_back(node_states[j]);
//                temp_feat.push_back(node_states[p.first]);

//                temp_feat.push_back(node_states[j]*node_states[p.first]);

//                if (p.second * node_states[j] * node_states[p.first] < 0.0)
//                {
//                    temp_feat.push_back(0.0);
//                }
//                else{
//                    temp_feat.push_back(1.0);
//                }   n n
                temp_edge_feat.push_back(1.0);
                edge_offset += 4;
                edge_cnt++;
                edge_feat.push_back(temp_edge_feat);
            }
            if (!actions)
            {
                rep_global->rowIndex.push_back(node_cnt + j);
                rep_global->colIndex.push_back(i);
                rep_global->value.push_back(1.0);
            }
        }

        if (actions)
        {
            auto act = actions[idxes[i]];
            assert(act >= 0 && act < g->num_nodes);
            act_select->rowIndex.push_back(i);
            act_select->colIndex.push_back(node_cnt + act);
            act_select->value.push_back(1.0);
        }
        node_cnt += g->num_nodes;
	}
    assert(edge_offset == edge_feat.size());
    assert(edge_cnt == (int)graph.num_edges);
    assert(node_cnt == (int)graph.num_nodes);

    n2nsum_param = n2n_construct(&graph,aggregatorID);
    e2nsum_param = e2n_construct(&graph);
    n2esum_param = n2e_construct(&graph);
    subgsum_param = subg_construct(&graph);

}
void PrepareBatchGraph::SetupTrain(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list,
                           std::vector< std::vector<int> > covered,
                           const int* actions,
                           int PE_dim)
{
    SetupGraphInput(idxes, g_list, covered, actions, PE_dim);
}
void PrepareBatchGraph::SetupPredAll(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list,
                           std::vector< std::vector<int> > covered,
                           int PE_dim)
{
    SetupGraphInput(idxes, g_list, covered, nullptr, PE_dim);
}
std::shared_ptr<sparseMatrix> n2n_construct(GraphStruct* graph, int aggregatorID)
{
    //aggregatorID = 0 sum
    //aggregatorID = 1 mean
    //aggregatorID = 2 GCN

    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_nodes;
    result->colNum = graph->num_nodes;


	for (unsigned int i = 0; i < graph->num_nodes; ++i)
	{
		auto& list = graph->in_edges->head[i];

		for (size_t j = 0; j < list.size(); ++j)
		{
		    switch(aggregatorID)
		    {
		       case 0:
		       {
		          result->value.push_back(1.0);
		          break;
		       }
		       case 1:
		       {
		          result->value.push_back(1.0/(double)list.size());
		          break;
		       }
		       case 2:
		       {
		          int neighborDegree = (int)graph->in_edges->head[list[j].second].size();
		          int selfDegree = (int)list.size();
		          double norm = sqrt((double)(neighborDegree+1))*sqrt((double)(selfDegree+1));
		          result->value.push_back(1.0/norm);
		          break;
		       }
		       default:
		       {
		          result->value.push_back(1.0);
		          break;
		       }
		    }

            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].second);


		}
	}
    return result;
}

std::shared_ptr<sparseMatrix> e2n_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_nodes;
    result->colNum = graph->num_edges;
	for (unsigned int i = 0; i < graph->num_nodes; ++i)
	{
		auto& list = graph->in_edges->head[i];
		for (size_t j = 0; j < list.size(); ++j)
		{
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].first);
		}
	}
    return result;
}

std::shared_ptr<sparseMatrix> n2e_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_edges;
    result->colNum = graph->num_nodes;

	for (unsigned int i = 0; i < graph->num_edges; ++i)
	{
        result->value.push_back(1.0);
        result->rowIndex.push_back(i);
        result->colIndex.push_back(graph->edge_list[i].first);
	}
    return result;
}

std::shared_ptr<sparseMatrix> e2e_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_edges;
    result->colNum = graph->num_edges;
    for (unsigned int i = 0; i < graph->num_edges; ++i)
    {
        int node_from = graph->edge_list[i].first, node_to = graph->edge_list[i].second;
        auto& list = graph->in_edges->head[node_from];
        for (size_t j = 0; j < list.size(); ++j)
        {
            if (list[j].second == node_to)
                continue;
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].first);
        }
    }
    return result;
}

std::shared_ptr<sparseMatrix> subg_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_subgraph;
    result->colNum = graph->num_nodes;
	for (unsigned int i = 0; i < graph->num_subgraph; ++i)
	{
		auto& list = graph->subgraph->head[i];

		for (size_t j = 0; j < list.size(); ++j)
		{
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j]);
		}
	}
    return result;
}
