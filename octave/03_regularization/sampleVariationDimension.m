% =============  =============  =============  =============  =============
% initialize octave env
clear; close all; clc

% =============  =============  =============  =============  =============
% load sample data
fprintf('\n\n* * * * * * * * * * * * * * * * * * * * * * * *\n');
fprintf('load data');
data = load('ex2data2.txt');

% show sample data
Xbase = data(:, [1,2]);
y = data(:, 3);

plotData(Xbase, y);

hold on;
xlabel('Test1');
ylabel('Test2');
legend('y=1', 'y=0');
hold off;

pause;

max_degree = 10;
results = [];
costs = zeros(max_degree, 1);
% =============  =============  =============  =============  =============
% variation Polynomial Demensions
for degree = [1:max_degree]
  % convert to Polynomial Features
  X = mapFeature(Xbase(:,1), Xbase(:,2), degree);

  [m, n] = size(X);

  fprintf('\n\n* * * * * * * * * * * * * * * * * * * * * * * *\n');
  fprintf('degree: %d\n',degree);
  fprintf('dataset size: %d\n', m);
  fprintf('data dimensions: %d\n', n);

  % =============  =============  =============  =============  =============
  % Optimizing
  lambda = 1.0;
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  
  initial_theta = zeros(n, 1);
  [theta, cost] = fminunc( @(t)(costFunctionWithRegularization(t, X, y, lambda)), initial_theta, options);

  % plot boundary
  plotDataDecisionBoundary(theta, X, y, degree);

  hold on;
  title(sprintf('Decision boundary (degree=%d / cost=%f)', degree, cost));
  xlabel('Test1');
  ylabel('Test2');
  legend('y=1', 'y=0');
  hold off;
  
  results(degree).polynomial = n;
  results(degree).cost = cost;
  results(degree).theta = theta;
  costs(degree) = cost;
end

% =============  =============  =============  =============  =============
% Output Summary

[minc, minic] = min(costs);
[maxc, maxic] = max(costs);

fprintf('%6s | %5s | %8s\n', 'degree', 'polys', 'cost');

for d = [1:max_degree]
  mark = '';
  if (minic == d)
    mark = '<minimal>';
  elseif (maxic == d)
    mark = '<MAXIMAL>';
  end
  res = results(d);
  fprintf('%6d | %5d | %f %s\n', ...
    d, res.polynomial, res.cost, mark);
end







