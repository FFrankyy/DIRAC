import argparse

def parsers():
    # Training settings
    parser = argparse.ArgumentParser(description='Hyperparameters for DIRAC')
    ## lattice generation
    parser.add_argument('--lattice_num_min', type=int, default=4, help='min scale of lattice generation')
    parser.add_argument('--lattice_num_max', type=int, default=5, help='max scale of lattice generation')
    parser.add_argument('--test_scale', type=int, default=10, help='scale of test instance')
    parser.add_argument('--cut_num', type=int, default=10000, help='calculate only cut_num instances')
    parser.add_argument('--stepRatio', type=float, default=0.01, help='stepRatio of each step')
    parser.add_argument('--g_type', type=str, default='lattice', help='max scale of lattice generation')
    parser.add_argument('--lattice_dim', type=int, default=3, help='dimension of lattice generation')
    parser.add_argument('--lattice_weight_distribution', type=str, default='normal', help='weight distribution of lattice generation')
    parser.add_argument('--lattice_Periodic', type=bool, default=True, help='whether is periodic for lattice generation')
    parser.add_argument('--gid', type=int, default=1, help='test graph id')
    ## DQN
    parser.add_argument('--gamma', type=float, default=1, help='control the weight of the long-term gain')
    parser.add_argument('--update_time', type=int, default=1000, help='update time to copy current Net params to target net')
    parser.add_argument('--embed_dim', type=int, default=64, help='node embedding dimension.')
    parser.add_argument('--reg_hidden', type=int, default=32, help='dimension of hidden neurons')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--max_bp_iter', type=int, default=5, help='max number of neighbor aggregations')
    parser.add_argument('--memory_size', type=int, default=50000, help='size of replay buffer memory')
    parser.add_argument('--max_iteration', type=int, default=1000000, help='max number of training iterations')
    parser.add_argument('--n_step', type=int, default=3, help='n-step Q updates')
    parser.add_argument('--node_feat_init', type=int, default=3, help='dimension of node input features')
    parser.add_argument('--edge_feat_init', type=int, default=4, help='dimension of edge input features')
    parser.add_argument('--PE_dim', type=int, default=10, help='dimension of position encoding')  #
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--n_valid', type=int, default=100, help='number of validation instances')
    parser.add_argument('--num_env', type=int, default=1, help='number of envs')
    parser.add_argument('--aggregatorID', type=int, default=0, help='0 sum  1 mean  2 GCN')
    parser.add_argument('--inf', type=float, default=float('inf'), help='infinite constants')
    parser.add_argument('--model_name', type=str, default='Grid_Search', help='denote the file name to store the trained models')

    args = parser.parse_args()
    return  args
