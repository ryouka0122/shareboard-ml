function J = computeCost(X, y, theta)
  
  % initialize valiables
  m = length(y);
  
  % compute cost
  J = sum( (X * theta - y) .^ 2 ) / (2*m);
  
end