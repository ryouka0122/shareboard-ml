% =============  =============  =============  =============  =============
% Initialization
clear; close all; clc

% =============  =============  =============  =============  =============
% Load data
load('ex8data1.mat');

[m, n] = size(X);

fprintf('dataset : %d\n', m);
fprintf('dimension : %d\n', n);


% =============  =============  =============  =============  =============
% Display dataset
plot(X(:,1), X(:,2), 'bx');
axis([0 35 0 35]);
xlabel('Latency (ms)');
ylabel('Thoughput (mb/s)');

pause;

% =============  =============  =============  =============  =============
% Estimate dataset statistics

[mu sigma2] = estimateGaussian(X);

p = multivariateGaussian(X, mu, sigma2);

visualizeFit(X, mu, sigma2);
xlabel('Latency (ms)');
ylabel('Thoughput (mb/s)');

pause;

% =============  =============  =============  =============  =============
% Estimate dataset statistics

pval = multivariateGaussian(Xval, mu, sigma2);

[bestThreshold histThreshold] = selectThreshold(yval, pval);

fprintf('epsilon : %e\n', bestThreshold.epsilon);
fprintf('F1 : %f\n', bestThreshold.F1);

outliers = find(p<bestThreshold.epsilon);

hold on;
plot(X(outliers, 1), X(outliers, 2), 'ro', ...
      'LineWidth', 2, ...
      'MarkerSize', 10);
hold off


% =============  =============  =============  =============  =============
% Pirint out history selectThreshold
printHistory(histThreshold, bestThreshold.index);