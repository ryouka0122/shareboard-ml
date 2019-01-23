function [J, grad] = costFunction(theta, X, y)
  m = length(y);
  
  hx = sigmoid(X*theta);
  
  J = sum(-y .* log(hx) .- (1-y) .* log(1-hx)) / m;
  
  grad = sum( (hx - y) .* X)' / m;
  
end