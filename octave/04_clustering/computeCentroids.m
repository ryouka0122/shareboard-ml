function centroids = computeCentroids(X, idx, K)
  
  % set varialbes (m-> dataset size / n-> degree at dataset)
  [m, n] = size(X);
  
  centroids = zeros(K, n);
  
  cnt = zeros(K, 1);
  
  for i = 1:m
    key = idx(i);
    centroids(key, :) += X(i, :);
    cnt(key) += 1;
  end
  
  centroids = centroids ./ cnt;
end