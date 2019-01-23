function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  
  % initialize 
  m = length(y);
  J_hisory = zeros(num_iters, 1);
  
  for iter = 1:num_iters
    % update theta
    theta = theta - alpha / m * X' * (X * theta - y);
    
    J_history(iter) = computeCost(X, y, theta);
    
  end
  
end
