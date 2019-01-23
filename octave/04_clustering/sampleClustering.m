% =============  =============  =============  =============  =============
% Initialization
clear ; close all; clc

% =============  =============  =============  =============  =============
% Load data

% set X from mat file.
load('ex7data2.mat');

% Select an initial set of centroids
K = 3; % 3 Centroids
initial_centroids = [3 3; 6 2; 8 5];

% Find the closest centroids for the examples using the
% initial_centroids
idx = findClosestCentroids(X, initial_centroids);

fprintf('Closest centroids for the first 3 examples: \n')
fprintf(' %d\n', idx(1:3));
pause;

% =============  =============  =============  =============  =============
% Compute closest centroids
centroids = computeCentroids(X, idx, K);

fprintf('Centroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');

pause;



% =============  =============  =============  =============  =============
% K-Means Clustering
fprintf('\nRunning K-Means clustering on example dataset.\n\n');

% Settings for running K-Means
load('ex7data2.mat');  % into X
K = 3;
max_iters = 10;

initial_centroids = [3 3; 6 2; 8 5];

% Run K-Means algorithm. The 'true' at the end tells our function to plot
% the progress of K-Means
[centroids, idx, cent_hist] = runKMeans(X, initial_centroids, max_iters, true);
fprintf('\nK-Means Done.\n\n');


% =============  =============  =============  =============  =============
% Output Summary
fprintf('%5s | %20s | %20s | %20s\n', ...
          'iters', 'centroid 1', 'centroid 2', 'centroid 3');

before_centroids = zeros(K, 2);
for i=1:max_iters
  centroid = cent_hist(i).centroid;
  fprintf('%5d |', i);
  tstr='';
  for j=1:K
    mark = ' ';
    if before_centroids(j, :) == centroid(j, :)
      mark = '*';
    end
    tstr=sprintf('%s |%s(%f, %f)', tstr, mark, centroid(j, 1), centroid(j, 2));
    before_centroids(j, :) = centroid(j, :);
  end
  fprintf('%s\n', substr(tstr, 3));
end



