function [centroids, idx, cent_hist] = runKMeans(X, initial_centroids, max_iters, plot_progress)
  
  if ~exist('plot_progress', 'var') || isempty(plot_progress)
    plot_progress = false;
  end
  
  if plot_progress
    figure;
    hold on;
  end
  
  [m n] = size(X);
  
  % initialize variables
  K = size(initial_centroids, 1);
  centroids = initial_centroids;
  previous_centroids = centroids;
  idx = zeros(m, 1);

  cent_hist = [];
  for i=1:max_iters
    
    % output progress
    fprintf('K-Means iteration %d/%d...\n', i, max_iters);
    if exist('OCTAVE_VERSION')
      fflush(stdout);
    end
    
    % for each example in X, assign it to the closest centroid
    idx = findClosestCentroids(X, centroids);
    
    if plot_progress
      plotProgressKMeans(X, centroids, previous_centroids, idx, K, i);
      previous_centroids = centroids;
      fprintf('Press enter to continue.\n');
      pause;
    end
    
    % given the memberships, compute new centroids
    centroids = computeCentroids(X, idx, K);
    cent_hist(i).centroid = centroids;
  end

  if plot_progress
    hold off;
  end

end