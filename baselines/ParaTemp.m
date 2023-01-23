% Input:
% JJJ: adjacency matrix Jij, size of NxN.
% tA: number of epochs
% seed: random seed, if set to be zero, then use default MATLAB random
% number generator.
% outfid and ifile only affect where the output message is written, can be
% modified if necessary.

% Output:
% egytrajs: the lowest energy in each epoch.
% timetrajs: time cost of each epoch.
function [egytrajs,timetrajs] = ParaTemp(JJJ,tA,seed,outfid,ifile)
%% Parameters
if seed~=0
    rng(seed);
end
nreps = 20;
tmin = 0.1;
tmax = 1.6;
ts = linspace(tmax,tmin,nreps);
nsites = size(JJJ,1);
states = sign(randn(nreps,nsites));
egys = zeros(nreps,1);
timetrajs = zeros(tA,1);
egytrajs = zeros(tA,1);
JJJ = sparse(JJJ);
%% Initialization
for i = 1:nreps
    states(i,:) = 2*randi(2,nsites,1)-3;
    egys(i) = -0.5*states(i,:)*JJJ*states(i,:)';
end
egytrajs(1) = min(egys);

%% PT_Variant_B
tic
for epoch = 1:tA
    for i = 1:nreps
        egys(i) = -0.5*states(i,:)*JJJ*states(i,:)';
        beta = 1/ts(i);
        for cycle = 1:nsites
            rnsite = randperm(nsites,1);
            site_energy = -states(i,rnsite)*JJJ(rnsite,:)*states(i,:)';
            delta_energy = -2*site_energy;
            if delta_energy<=0 || (exp(-delta_energy*beta)>rand())
                egys(i) = egys(i) + delta_energy;
                states(i,rnsite) = -states(i,rnsite);
                if egys(i)<egytrajs(epoch)
                    egytrajs(epoch) = egys(i);
                end
            end
        end
    end
    i = randperm(nreps-1,1)+1;
    if i>1
        delta = (1/ts(i-1)-1/ts(i))*(egys(i)-egys(i-1));
        if delta<=0 || (exp(-delta)>rand())
            tempegy = egys(i);
            egys(i) = egys(i-1);
            egys(i-1) = tempegy;
            tempstate = states(i,:);
            states(i,:) = states(i-1,:);
            states(i-1,:) = tempstate; 
        end
    end
    if epoch>1
    egytrajs(epoch) = min(egytrajs(epoch-1),egytrajs(epoch));
    end
    timetrajs(epoch) = toc;
    fprintf(outfid,'Graph: %d, epoch: %d, result: %.10f, time: %.10f\n' ...
            ,ifile,epoch,egytrajs(epoch)/size(JJJ,1),timetrajs(epoch));
end