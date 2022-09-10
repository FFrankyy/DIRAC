#ifndef cfg_H
#define cfg_H

#include <iostream>
#include <cstring>
#include <fstream>
#include <set>
#include <map>

typedef float Dtype;

struct cfg
{
    static int embed_dim;
    static int batch_size;
    static int max_n, min_n;
    static int mem_size;
    static int node_dim;
    static int aux_dim;
    static bool msg_average; 

    static void LoadParams(const int argc, const char** argv)
    {
        for (int i = 1; i < argc; i += 2)
        {
            if (strcmp(argv[i], "-embed_dim") == 0)
                embed_dim = atoi(argv[i + 1]); 
            if (strcmp(argv[i], "-max_n") == 0)
                max_n = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-min_n") == 0)
                min_n = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-mem_size") == 0)
                mem_size = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-batch_size") == 0)
                batch_size = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-msg_average") == 0)
                msg_average = atoi(argv[i + 1]);                 

        }

        if (n_step <= 0)
            n_step = max_n;

        std::cerr << "mem_size = " << mem_size << std::endl;
        std::cerr << "min_n = " << min_n << std::endl;
        std::cerr << "max_n = " << max_n << std::endl;
        std::cerr << "batch_size = " << batch_size << std::endl;        
        std::cerr << "embed_dim = " << embed_dim << std::endl;        
        std::cerr << "w_scale = " << w_scale << std::endl;
        std::cerr << "msg_average = " << msg_average << std::endl;
    }
};

#endif
