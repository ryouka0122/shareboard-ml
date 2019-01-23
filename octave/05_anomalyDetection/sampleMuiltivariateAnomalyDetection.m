% =============  =============  =============  =============  =============
% Initialization
clear; close all; clc

% =============  =============  =============  =============  =============
% Load data
load('ex8data2.mat');

[m, n] = size(X);
fprintf('** Learning dataset **\n');
fprintf(' dataset : %d\n', m);
fprintf(' dimension : %d\n\n', n);

[mval, nval] = size(Xval);
fprintf('** CrossVaridation dataset **\n');
fprintf(' dataset : %d\n', mval);
fprintf(' dimension : %d\n\n', nval);


% =============  =============  =============  =============  =============
% Estimate dataset statistics

[mu sigma2] = estimateGaussian(X);
p = multivariateGaussian(X, mu, sigma2);


% =============  =============  =============  =============  =============
% Estimate dataset statistics

pval = multivariateGaussian(Xval, mu, sigma2);

[bestThreshold, histThreshold] = selectThreshold(yval, pval);

outliers = sum(p<bestThreshold.epsilon);

fprintf('epsilon : %e\n', bestThreshold.epsilon);
fprintf('F1 : %f\n', bestThreshold.F1);
fprintf('Outliers found : %d\n\n', outliers);


% =============  =============  =============  =============  =============
% Pirint out history selectThreshold
printHistory(histThreshold, bestThreshold.index);
