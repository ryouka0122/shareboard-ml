% =============  =============  =============  =============  =============
% initialize octave env
clear; close all; clc

% =============  =============  =============  =============  =============
% load sample data
fprintf('\n\n* * * * * * * * * * * * * * * * * * * * * * * *\n');
fprintf('load data');
data = load('ex2data1.txt');
% data
pause;

% =============  =============  =============  =============  =============
% show sample data
X = data(:, [1,2]);
y = data(:, 3);

plotData(X, y);

hold on;
xlabel('Exam 1 score');
ylabel('Exam 2 score');
legend('Admitted', 'Not admitted');
hold off;

pause;


[m, n] = size(X);
X = [ones(m,1) X];

fprintf('\n\n* * * * * * * * * * * * * * * * * * * * * * * *\n');
fprintf('dataset size: %d\n', m);
fprintf('data dimensions: %d\n', n);
pause;

% =============  =============  =============  =============  =============
% Compute cost
fprintf('\n\n* * * * * * * * * * * * * * * * * * * * * * * *\n');
initial_theta = zeros(n+1, 1);
[cost, grad] = costFunction(initial_theta, X, y);
fprintf('[at initial theta(zeros)]\n');
fprintf('cost:\n  %f\n', cost);
fprintf('gradient:\n');
fprintf('  %f \n', grad);
fprintf('\n');

test_theta = [-24; 0.2; 0.2];
[cost, grad] = costFunction(test_theta, X, y);
fprintf('[at test theta(-24.0, 0.2, 0.2)]\n');
fprintf('cost:\n  %f\n', cost);
fprintf('gradient:\n');
fprintf('  %f \n', grad);
fprintf('\n');


pause;

% =============  =============  =============  =============  =============
% Optimizing using fminunc
% function fminunc is built-in function.

options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = fminunc( @(t)(costFunction(t, X, y)), initial_theta, options);

fprintf('[Optimized theta]\n');
fprintf('cost:\n  %f\n', cost);
fprintf('theta:\n');
fprintf('  %f \n', theta);
fprintf('\n');

% plot boundary
plotDataDecisionBoundary(theta, X, y);

hold on;
xlabel('Exam 1 score');
ylabel('Exam 2 score');
legend('Admitted', 'Not admitted');
hold off;






