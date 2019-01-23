function [J, grad] = costFunctionWithRegularization(theta, X, y, lambda)
  m = length(y);
  
  hx = sigmoid(X*theta);
  
  reg_theta = theta;
  reg_theta(1) = 0;
  
  J = sum(-y .* log(hx) .- (1-y) .* log(1-hx)) / m ...    % cost term
        + lambda / (2*m) * sum(reg_theta .^ 2);           % regularization term
  
  grad = sum( (hx - y) .* X)' / m ...                     % cost term
        + lambda * reg_theta / m;                         % regularization term
  
end