% =============  =============  =============  =============  =============
% initialize octave env
clear; close all; clc

% =============  =============  =============  =============  =============
% load sample data
fprintf('\n\n* * * * * * * * * * * * * * * * * * * * * * * *\n');
fprintf('load data');
data = load('ex2data2.txt');
% data
pause;

% =============  =============  =============  =============  =============
% show sample data
X = data(:, [1,2]);
y = data(:, 3);

plotData(X, y);

hold on;
xlabel('Test1');
ylabel('Test2');
legend('y=1', 'y=0');
hold off;

pause;

% convert to Polynomial Features
degree = 6;
X = mapFeature(X(:,1), X(:,2), degree);

[m, n] = size(X);

fprintf('\n\n* * * * * * * * * * * * * * * * * * * * * * * *\n');
fprintf('dataset size: %d\n', m);
fprintf('data dimensions: %d\n', n);
pause;

% =============  =============  =============  =============  =============
% Compute cost
fprintf('\n\n* * * * * * * * * * * * * * * * * * * * * * * *\n');
initial_theta = zeros(n, 1);
[cost, grad] = costFunctionWithRegularization(initial_theta, X, y, 1);
fprintf('[at initial theta(all-zeros with lambda=1)]\n');
fprintf('cost:\n  %f\n', cost);
fprintf('gradient:\n');
fprintf('  %f \n', grad);
fprintf('\n');

test_theta = ones(n, 1);
[cost, grad] = costFunctionWithRegularization(test_theta, X, y, 10);
fprintf('[at test theta(all-ones with lambda=10)\n');
fprintf('cost:\n  %f\n', cost);
fprintf('gradient:\n');
fprintf('  %f \n', grad);
fprintf('\n');


pause;

% =============  =============  =============  =============  =============
% Optimizing using fminunc
% function fminunc is built-in function.
lambda = 1;
options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = fminunc( @(t)(costFunctionWithRegularization(t, X, y, lambda)), initial_theta, options);

fprintf('[Optimized theta]\n');
fprintf('cost:\n  %f\n', cost);
fprintf('theta:\n');
fprintf('  %f \n', theta);
fprintf('\n');

% plot boundary
plotDataDecisionBoundary(theta, X, y, degree);

hold on;
xlabel('Test1');
ylabel('Test2');
legend('y=1', 'y=0');
hold off;






