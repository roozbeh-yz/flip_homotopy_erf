%% find the sigma on each level that makes the Jacobians bi-Lipschitz
%%  for an individual input
% written by Roozbeh Yousefzadeh, last modified Oct. 30, 2018.

% 1- x is the individual input
% 2- net is the trained neural network defined in readTrainedNN.m
% 3- g_thold is the threshold for gradient of output - sigmaNN will be tuned
%    such that all the outputs on a layer have greadients greater than g_thold
% 4- id is a scalar id associated with x, for printing or identification
%    purposes
% 5- sigma_all is the set of sigmas that make all the derivatives >= g_thold
% 6- sigma_mean is the set of sigmas that make the derivative for 
%    mean of outputs on each layer >= g_thold

function [sigma_all,sigma_mean] = sigma_point(x,net,g_thold,id)
if nargin == 4
    print_warning = 1;
else
    print_warning = 0;
end
% thold stands for threshold
% g_thold is the threshold for gradients of the output of nodes
% p_thold is the point on error function that has derivative g_thold
temp = x * net.w{1};
sigma_mean = zeros(1,net.nl); sigma_all = sigma_mean;

p_thold = sqrt(-log(sqrt(pi)*g_thold/2)); % z = x/sigma

% find the sigma for the first layer
% mean ensures mean of the inputs to a layer is
sigma_mean(1) = mean(abs(temp)) ./ p_thold;
sigma_all(1) = max(abs(temp)) ./ p_thold;

y_mean{1} = erf( (x * net.w{1}) ./ sigma_mean(1) );
y_all{1}  = erf( (x * net.w{1}) ./ sigma_all(1) );

for i = 2:net.nl
    temp_mean = y_mean{i-1} * net.w{i};
    temp_all = y_all{i-1} * net.w{i};

    sigma_mean(i) = mean(abs(temp_mean)) ./ p_thold;
    sigma_all(i) = max(abs(temp_all)) ./ p_thold;
    
    y_mean{i} = erf( (y_mean{i-1} * net.w{i}) ./ sigma_mean(i) );
    y_all{i} = erf( (y_all{i-1} * net.w{i}) ./ sigma_all(i) );
end

flat_thold = 2/sqrt(pi)/g_thold;
if sum(sigma_all>(flat_thold)) > 0 && print_warning
    fprintf('Warning! Some fo the sigmas are too large for input id %d.\n', id);
end

end
