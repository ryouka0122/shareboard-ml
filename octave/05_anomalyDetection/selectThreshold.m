function [bestThreshold histThreshold] = selectThreshold(yval, pval)
  % number of test step
  step = 1000;
  % number of history
  history_size = 1000;
  
  % initialize variables
  histThreshold = [];
  
  bestThreshold.index = 0;
  bestThreshold.epsilon = 0;
  bestThreshold.F1 = 0;

  % calculate variables
  stride = floor(step / history_size);
  stepsize = (max(pval) - min(pval)) / 1000;

  % counter for history
  i = 1;
  for epsilon = min(pval):stepsize:max(pval)
    predictions = (pval < epsilon);
    
    TP = sum( (predictions==1) & (yval==1) );
    FP = sum( (predictions==1) & (yval==0) );
    FN = sum( (predictions==0) & (yval==1) );

    % Guard from divided by zero
    prec = 0;
    if (TP + FP) != 0
      prec = TP / (TP + FP);
    end
    
    % Guard from divided by zero
    rec = 0;
    if (TP + FN) != 0
      rec = TP / (TP + FN);
    end
    
    % Guard from divided by zero
    F1=0;
    if (prec + rec) != 0
      F1 = (2.0 * prec * rec) / (prec + rec);
    end
    
    if F1 > bestThreshold.F1
      % update best value
      bestThreshold.F1 = F1;
      bestThreshold.epsilon = epsilon;
      bestThreshold.index = i;
    end
    
    % set value into history
    if mod(i, stride) == 0
      idx = length(histThreshold) + 1;
      histThreshold(idx).epsilon = epsilon;
      histThreshold(idx).F1 = F1;
    end
    
    ++i; % increase counter
  end
  
end