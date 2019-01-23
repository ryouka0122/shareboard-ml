function plotProgressKMeans(X, centroids, previous, idx, K, i)
  
  % plot example data
  plotDataPoints(X, idx, K);
  
  % plot centroids
  plot(centroids(:,1), centroids(:,2), 'x', ...
        'MarkerEdgeColor', 'k', ...
        'MarkerSize', 10, ...
        'LineWidth', 3);
  
  % draw the history of the centroids
  for j = 1:size(centroids, 1)
    drawLine(centroids(j, :), previous(j, :));
  end
  
  % title
  title(sprintf('Iteration number %d', i));
  
end
