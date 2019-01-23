function plotData(x, y, x_lbl, y_lbl)
  
  figure;
  
  plot(x, y, 'rx', 'MarkerSize', 10);
  
  ylabel(y_lbl);
  xlabel(x_lbl);
end
