function idx = findClosestCentroids(X, centroids)
  
  % get number of cluster 
  K = size(centroids);
  
  % initialize
  idx = zeros(size(X, 1), 1);
  
  for i = 1:size(X, 1)
    dist = zeros(K, 1);
    pos = X(i, :);
    
    for c = 1:K
      dist(c) = sum((pos - centroids(c, :)) .^ 2);
    end
    [d, j] = min(dist);
    idx(i) = j;
  end
end

