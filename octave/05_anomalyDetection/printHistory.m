function printHistory(histThreshold, bestIndex)

  % =============  =============  =============  =============  =============
  % define constants
  hdtlRange = 3;
  starRange = 2;

  % =============  =============  =============  =============  =============
  % initialize variables
  histLen = length(histThreshold);

  % head index range
  hb = 1;
  he = hdtlRange;
  
  % tail index range
  tb = histLen - hdtlRange + 1;
  te = histLen;
  
  % best index range
  star = bestIndex;
  sb = star - starRange + 1;
  se = star + starRange;
  
  % flagment for print skip text
  skipHead = false;
  skipTail = false;
  
  % define output skip text
  printSkipText = @()(fprintf('%5s\n', ':'));

  % =============  =============  =============  =============  =============
  % Print history list
  
  % print header
  fprintf('%5s | %13s | %8s\n', 'index', 'epsilon', 'F1');
  
  % print contents
  for i = [1:histLen]
    threshold = histThreshold(i);
    
    % judge bestIndex
    mark = '';
    if i == star
      mark = '*';
    end
    
    % judge print data
    if (hb <= i && i <= he) || (sb <= i && i <= se) || (tb <= i && i <= te)
      fprintf('%5d | %e | %f %s\n', i, threshold.epsilon, threshold.F1, mark);
    elseif (skipHead == false) && (he < i && i < sb)
      % for between head range and best range
      printSkipText();
      skipHead = true;
    elseif (skipTail == false) && (se < i && i < tb)
      % for between best range and tail range
      printSkipText();
      skipTail = true;
    end
    
  end

end