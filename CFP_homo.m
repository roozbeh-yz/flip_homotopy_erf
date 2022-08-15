%% FindClosestFlipPoint via homotopy and NLopt
% written by Roozbeh Yousefzadeh, last modified July 21, 2019.

% This function changes the sigma of the neural network from sigmaNN to sigma_homo
% and then transforms it back to original sigmaNN through niter iterations,
% at each iteration it uses the CFP function to find the closest flip point
% and then uses that as the starting point for the next iteration

% inputs
% 1- x: is a single input to the network
% 2- this: is the class associated with x
% 3- that: is the class that we want to flip to
% 4- net is a struct that contains the trained NN
% 5- itertime is the maximum time limit for the NLopt to search at each 
%    homotopy iteration
% 6- niter is the number of iterations for homotopy transformation
% 7- sigma_homo is the sigma for the transformed state of NN
% 8- x0 the starting point - if not provided it will be set equal to x
% 9- bounds: a struct containing two scalars or vectors same size of x -
%       bounds.low defines the lower bound and bounds. up defines the upper
%       if not provided, the lower bound will be zero and upper bound 1e4 -
%       not defined yet

% outputs
% xf is the closest flip point
% c is the softmax output that has been equalized for this and that classes
% c will be NaN if the soft output of classes is unequal
% dist is the Euclidean distance between the x_hat and its closest flip point
% elap_time is the total time spent to find the xf

function [xf,c,dist,elap_time] = CFP_homo(x,this,that,net,itertime,niter,tau,x0,bounds)
% homo_options
thomo = tic;
if nargin < 8; x0 = x; end
if nargin < 9; bounds = []; end
if nargin < 10; b = 0; end
xf = x0;
net2 = net;
sigma_homo = sigma_point(x0,net,tau);
sdif = (sigma_homo - net.sigmaNN);
net2.sigmaNN = sigma_homo;
b_homo = trans_b(x0,this,that,net2);
bdif = b_homo - net.b{end};
for i = 0:niter
    fprintf(' ------ i = %d - \n', i);
    net2.sigmaNN = net.sigmaNN + sdif .* (1 - i/niter)^1.0;
    net2.b{end} = net.b{end} + bdif .* (1 - i/niter)^1.0;
    if i < niter
        [xf,c,dist,~] = CFP(x,this,that,net2,itertime,0,[],xf,bounds);
    else
        [xf,c,dist,~] = CFP(x,this,that,net2,itertime*3,0,[],xf,bounds);
    end
    ntry = 0;
    if isnan(c) && ntry > 0
        j = 0;
        while isnan(c) && j < ntry
            fprintf(' j = %d - i = %d  - homotopy iteration\n', j,i);
            j = j + 1;
            [xf2,c,dist,~] = CFP(x,this,that,net2,itertime*j*2,0,[],xf,bounds);
        end
        xf = xf2;
    end
end
elap_time = toc(thomo);
end


function [b_homo] = trans_b(x,this,that,net)
pdata = pointEval(x,net,0,0,0);
% pdata = calc_output(x,net);
yw = pdata.y{end} * net.w{end};

b = net.b{end};
b_homo = b;
fun = @(b_homo)fung(b_homo,b);
nonlcon = @(b_homo)mycon(b_homo,yw,this,that);
lb = [];  ub = [];
options = optimoptions('fmincon','Display','none','Algorithm','interior-point',... %interior-point
    'SpecifyObjectiveGradient',true, 'ConstraintTolerance', 1e-6, ...
    'HonorBounds',false,...
    'SpecifyConstraintGradient',true,'MaxIterations',500);
%     'OptimalityTolerance',1e-6, 'FunctionTolerance',1e-12,...

[b_homo,fval,retcode,output] = fmincon(fun,b,[],[],[],[],lb,ub,nonlcon,options);
pdata.z;
net.b{end} = b_homo;
pdata = pointEval(x,net,0,0,0);
pdata.z;
end

function [f,g] = fung(b_homo,b)
f = norm(b-b_homo,2)^2;
g = 2.*(b-b_homo);
end

function [c,ceq,gc,gceq] = mycon(b_homo,yw,this,that)
y = yw + b_homo;
nb = length(b_homo);

ceq = y(this) - y(that);
gceq = zeros(size(b_homo))';
gceq(this) = 1;  gceq(that) = -1;

c = [];  gc = [];
if length(b_homo) > 2
    otherList = setdiff(1:length(b_homo),[this,that]);
    I = eye(nb);
    c = 1.1 * y(otherList) - y(this);
    gc = 1.1 .* I(:,otherList) - ...
        sparse(1,this,1,1,nb)' * ones(1,length(otherList));
end

end
