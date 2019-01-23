% initialize octave env
clear; close all; clc

% load sample data
fprintf('\n\n* * * * * * * * * * * * * * * * * * * * * * * *\n');
fprintf('load data');
pause;

data = load('ex1data1.txt');
% data
pause;

% show sample data
X = data(:, 1);
y = data(:, 2);
plotData(X, y, 'Population of City in 10,000s', 'Profit in $10,000s');

pause;

m = length(y);
fprintf('\n\n* * * * * * * * * * * * * * * * * * * * * * * *\n');
fprintf('dataset size: %d\n', m);
pause;

% ---- gradient descent --------------------------------------

% initialize dataset for gradient descent
X = [ones(m, 1), data(:, 1)];
theta = zeros(2,1);

fprintf('\n\n* * * * * * * * * * * * * * * * * * * * * * * *\n');
fprintf('Compute cost value\n');
fprintf('theta = [ 0; 0] -> %f\n', computeCost(X, y, [ 0; 0]));
fprintf('theta = [-1; 2] -> %f\n', computeCost(X, y, [-1; 2]));

pause;

% settings gradient descent parameters
iterations = 1500;
alpha = 0.01;

% run gradient descent algorithm
theta = gradientDescent(X, y, theta, alpha, iterations);
fprintf('\n\n* * * * * * * * * * * * * * * * * * * * * * * *\n');
fprintf('optimized parameter theta using gradient descent\n');
fprintf('theta = \n');
fprintf('%f\n', theta);

pause;

% predict values
inputs = [ ...
   3.5; ...
   7.0; ...
  10.0; ...
  12.0; ...
  15.0; ...
  20.0
];

psize = length(inputs);

predicts = [ones(psize, 1), inputs] * theta;

% predict 

printf('predict value\n');
for i=1:psize
  fprintf('For Population = %d, predict a profit of %f\n', inputs(i)*10000, predicts(i));
end

%predict1 = [1, 3.5] * theta;
%fprintf('For Population = 35,000, predict a profit of &f\n', predict1 * 10000);
%predict2 = [1, 7.0] * theta;
%fprintf('For Population = 70,000, predict a profit of &f\n', predict2 * 10000);

hold on;
plot(X(:,2), X*theta, '-')
plot(inputs, predicts, '*k', 'MarkerSize', 10, 'LineWidth', 3)
legend('Training data ', 'Linear Regression', 'Predict')
grid on
hold off


% ------------------------------------------------------------------------------
% visualizing cot J

% grid area
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace( -1,  4, 100);

J_vals = zeros(length(theta0_vals), length(theta1_vals));

for i = 1:length(theta0_vals)
  for j = 1:length(theta1_vals)
    t = [theta0_vals(i) ; theta1_vals(j)];
    J_vals(i, j) = computeCost(X, y, t);
  end
end

% flip axis for meshgrid
J_vals = J_vals';

figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0');
ylabel('\theta_1');

figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0');
ylabel('\theta_1');
hold on;

% plot optimized theta value
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

