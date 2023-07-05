dim = 3;
scale = 30;
gid = 3;
embed_dim = 64;
max_bp_iter = 5;
batch_ratio = 0.01;
paras = load_3Dparas;
gfilename = sprintf("./data/%dD/%d/%d.txt",dim,scale,gid);
JJJ = gfile2smat(gfilename);

EITI = JJJ;
nsites = size(JJJ,1);
batch_size = round(nsites*batch_ratio);
nedges = nsites*6;
[rows,cols] = find(JJJ~=0);
edge_input = zeros(nsites*6,4);
edge_input(:,4) = 1;
EIT = zeros(nedges,2);
% neibs = zeros(nsites,dim*2);
neibs = cell(nsites,1);
neibsindex = cell(nsites,1);
for i = 1:nsites
    neibsindex{i} = rows((i-1)*dim*2+1:i*dim*2);
    neibs{i} = JJJ(i,rows((i-1)*dim*2+1:i*dim*2));
end
e2nsum_param = spalloc(nsites,nedges,nedges);%zeros(nsites,nedges);
n2esum_param = spalloc(nedges,nsites,nedges);%zeros(nedges,nsites);
count = 0;
for i = 1:length(rows)
    left = rows(i);
    right = cols(i);
    count = count+1;
    e2nsum_param(right,count) = 1; %This matrix and below can be used for all lattices with same size.
    n2esum_param(count,left) = 1;
    EITI(left,right) = count;
    EIT(count,:) = [left,right];
end

e2nsum_param = sparse(e2nsum_param);
n2esum_param = sparse(n2esum_param);
subgsum_param = ones(1,nsites);
rep_global = ones(nsites,1);
covered = zeros(nsites,1);
cur_node_embed = zeros(nsites,embed_dim);
%%

tic
count = 0;
for i = 1:length(rows)
    left = rows(i);
    right = cols(i);
    count = count+1;
    edge_input(count,1) = full(JJJ(left,right));
end
edge_input_0 = edge_input;
traj = [];
cur_node_embed = gpuArray(cur_node_embed);
edge_input = gpuArray(edge_input);
rep_global = gpuArray(rep_global);
e2nsum_param = gpuArray(e2nsum_param);
n2esum_param = gpuArray(n2esum_param);
subgsum_param = gpuArray(subgsum_param);
%%
while sum(covered)<nsites
cur_node_embed = zeros(nsites,embed_dim);
edge_input(:,2) = covered(EIT(:,1));
edge_input(:,3) = mod(covered(EIT(:,1))+covered(EIT(:,2)),2);
edge_init = edge_input*paras.w_e2l;
lv = 0;
%%
while lv<max_bp_iter
    lv = lv+1;
    % Edge Embed
    cur_node_embed_prev = cur_node_embed;
    msg_linear_node = cur_node_embed*paras.p_node_conv1;
    n2e = n2esum_param*msg_linear_node;
    n2e_linear = [n2e*paras.trans_edge_1,edge_init*paras.trans_edge_2];
    cur_edge_embed = myrelu(n2e_linear);
    %cur_edge_embed = gpuArray(cur_edge_embed);
    cur_edge_embed = mynormr(cur_edge_embed);
    % Node Embed
    e2n = e2nsum_param*cur_edge_embed;
    node_linear = [e2n*paras.trans_node_1,cur_node_embed*paras.trans_node_2];
    cur_node_embed = myrelu(node_linear);
    cur_node_embed = mynormr(cur_node_embed);
    cur_node_embed = [cur_node_embed,cur_node_embed_prev]*paras.w_l;
end

%cur_node_embed = mynormr(cur_node_embed);
y_potential = subgsum_param*cur_node_embed;
rep_y = rep_global*y_potential;
embed_s_a_all = [rep_y*paras.h11_weight,cur_node_embed*paras.h12_weight];
hidden1 = myrelu(embed_s_a_all);
hidden2 = myrelu(hidden1*paras.h2_weight);
last_output = hidden2;
q_on_all = last_output*paras.last_w;
q_on_all(covered==1) = -inf;
[values,maxindices] = maxk(q_on_all,batch_size);
covered(maxindices) = 1;
traj = [traj;maxindices];
end
toc
traj = gather(traj);
state = ones(nsites,1);
best_egy = +inf;
best_state = state;
egy = -0.5*state'*JJJ*state;
for i = 1:nsites
    isite = traj(i);
    site_energy = -state(isite)*(neibs{isite}*state(neibsindex{isite}));
    %site_energy = -state(isite)*(neibs(isite,:)*state(neibsindex(isite,:)));
    egy = egy-2*site_energy;
    state(isite) = -state(isite);
    if egy<best_egy
        best_egy = egy;
        best_state = state;
    end
end
toc
