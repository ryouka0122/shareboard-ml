function plotDataPoints(X, idx, K)
  
  % create palette
  palette = hsv(K+1);
  colors = palette(idx, :);
  
  % plot the data
  scatter(X(:, 1), X(:, 2), 15, colors);
  
end