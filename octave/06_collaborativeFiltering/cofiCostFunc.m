function [J, grad] = cofiCostFunc(params, Y, R, ...
                                  num_users, num_movies, num_features, ...
                                  lambda)
  
  % =============  =============  =============  =============  =============
  % Unfold params
  X = reshape(params(1:num_movies*num_features), num_movies, num_features);
  Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features);

  % =============  =============  =============  =============  =============
  % Initialize
  J = 0;
  X_grad = zeros(size(X));
  Theta_grad = zeros(size(Theta));

  % =============  =============  =============  =============  =============
  % Compute Cost J and gradient
  J = 1/2 * sum( sum( (R==1) .* (X*Theta' - Y) .^ 2 ) ) + ...
      lambda/2 * sum( sum(Theta.^2) ) + lambda/2 * sum( sum(X.^2) );

  for i = 1:num_movies
    X_grad(i,:) = R(i,:) .* (X(i,:) * Theta' - Y(i,:)) * Theta + ...
                  lambda * X(i,:);
  end
  
  for j = 1:num_users
    Theta_grad(j,:) = ( (X*Theta(j,:)' - Y(:,j)) .* R(:,j) )' * X + ...
                      lambda * Theta(j, :);
  end

  %for i = 1:num_movies
  %  for k = 1:num_features
  %    for j = 1:num_users
  %      X_grad(i,k) = X_grad(i,k) + R(i,j) * (Theta(j,k) * X(i,k) - Y(i,j)) * Theta(j,k);
  %      Theta_grad(j,k) = Theta_grad(j,k) + R(i,j) * (Theta(j,k) * X(i,k) - Y(i,j)) * X(i,k);
  %    end
  %  end
  %end

  grad = [X_grad(:); Theta_grad(:)];
end