function movieList = loadMovieList()
  
  % Initialize
  n = 1682;
  moveiList = cell(n, 1);

  % open file
  fid = fopen('movie_ids.txt');
  
  for i=1:n
    % read line
    line = fgets(fid);
    % split text
    [idx, movieName] = strtok(line, ' ');
    % set movie name
    movieList{i} = strtrim(movieName);
  end
  
  % close file handle
  fclose(fid);
end