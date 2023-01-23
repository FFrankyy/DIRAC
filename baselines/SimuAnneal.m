% Input:
% model: adjacency matrix Jij, size of NxN.
% ini_state: the initial state as a vector of Nx1.
function outmessage = SimuuAnneal(model,ini_state)

tinside = tic;

nsites = size(model,1);
dim = sum(sum(model~=0))/nsites/2;

%% Parameters.
NT = 500;
NS = 10;
betamin = 0.001;
betamax = 5.000;
betas = linspace(betamin,betamax,NT);

%% Initialize the data strctures.
lps = sparse(sign(abs(model)));         %store the link products sisj.
neibs = zeros(nsites,2*dim);            %store the neighbours' indices.
[rows cols] = find(model~=0);
for i = 1:nsites
    neib_indices = find(rows==i);
    neibs(i,:) = cols(neib_indices);
end
for i = 1:length(rows)
    left = rows(i);
    right = cols(i);
    lps(left,right) = ini_state(left)*ini_state(right);
end

%% Iterations.
lowest_energy = +100.0;
current_energy = -(1/2)*sum(sum(lps.*model));
for step = 1:NT
	T = 1/betas(step);
    for i = 1:nsites*NS
        rnindex = floor(nsites*rand())+1;
        rnneibs = neibs(rnindex,:);
        delta = 2*lps(rnindex,rnneibs)*model(rnindex,rnneibs)';
        if delta <= 0 || exp(-delta/T)>rand()
            lps(rnindex,rnneibs) = -lps(rnindex,rnneibs);
            lps(rnneibs,rnindex) = -lps(rnneibs,rnindex);
            current_energy = current_energy + delta;
        end
        if current_energy < lowest_energy
			lowest_energy = current_energy;
		end
    end
 end
no_improve = lowest_energy;

%% Local improve.
%stopCondition = false;
%while ~stopCondition
%    best_delta = 0;
%    best_site = 0;
%    for i = 1:nsites
%        ineibs = neibs(i,:);
%        delta = 2*lps(i,ineibs)*model(i,ineibs)';
%        if delta<=0 && delta<best_delta
%            best_delta = delta;
%            best_site = i;
%        end
%    end
%    if best_delta == 0
%        stopCondition = true;
%        break;
%    end
%    bestneibs = neibs(best_site,:);
%    lps(best_site,bestneibs) = -lps(best_site,bestneibs);
%    lps(bestneibs,best_site) = -lps(bestneibs,best_site);
%    current_energy = current_energy + best_delta;
%end




%% Finalize.
timecost = toc(tinside);
outmessage = ['result: ' ,num2str(-no_improve/nsites,18),', time:',num2str(timecost)];
end