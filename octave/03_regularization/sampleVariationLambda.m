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
% Optimizing on variable lambdas
options = optimset('GradObj', 'on', 'MaxIter', 400);

for lambda = [0.0, 0.3, 1.0, 3.0, 10.0, 100.0]
  initial_theta = zeros(n, 1);
  [theta, cost] = fminunc( @(t)(costFunctionWithRegularization(t, X, y, lambda)), initial_theta, options);

  % plot boundary
  plotDataDecisionBoundary(theta, X, y, degree);

  hold on;
  title(sprintf('Decision boundary (lambda=%f / cost=%f)', lambda, cost));
  xlabel('Test1');
  ylabel('Test2');
  legend('y=1', 'y=0');
  hold off;
end

