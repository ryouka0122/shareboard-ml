function [Ynorm Ymean] = normalizeRatings(Y, R)
  % initialize
  [m, n] = size(Y);
  Ymean = zeros(m, 1);
  Ynorm = zeros(size(Y));
  
  % Ynorm := Y - Ymean for rating movie
  for i = 1:m
    idx = find(R(i,:)==1);
    Ymean(i) = mean(Y(i, idx));
    Ynorm(i, idx) = Y(i, idx) - Ymean(i);
  end
  
end