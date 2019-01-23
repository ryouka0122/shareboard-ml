% =============  =============  =============  =============  =============
% Initialization
clear; close all; clc

showImage = false;

% =============  =============  =============  =============  =============
% define function PRINTINFO
function printInfo(mat, lbl)
  [m n] = size(mat);
  fprintf('** %s dataset **\n', lbl);
  fprintf(' dataset : %d\n', m);
  fprintf(' dimension : %d\n\n', n);
endfunction


% =============  =============  =============  =============  =============
% Load data into R and Y
load('ex8_movies.mat');

printInfo(R, 'R');
printInfo(Y, 'Y');

pause;


fprintf('Average rating for movie 1(Toy Story): %f / 5\n\n', ...
          mean(Y(1, R(1, :))));

% visualize ratings
if showImage
  imagesc(Y);
  ylabel('Movies');
  xlabel('Users');
end

% =============  =============  =============  =============  =============
% Load parameters into Theta, X, num_features, num_movies and num_users
load('ex8_movieParams.mat');

printInfo(Theta, 'Theta');
printInfo(X, 'X');
fprintf('num_features : %d\n', num_features);
fprintf('num_movies : %d\n', num_movies);
fprintf('num_users : %d\n', num_users);
fprintf('\n\n');
pause;


% =============  =============  =============  =============  =============
% New user dataset

movieList = loadMovieList();
my_ratings = zeros(1682, 1);


% rate movie 1 (Toy Story) "4"
my_ratings(1) = 4;
% rate movie 98 (Silence of the Lambs) "2"
my_ratings(98)= 2;

% rate other movies like / did not like
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;

fprintf('*** new user ratings ***\n');
for i = 1:length(my_ratings)
  if my_ratings(i) > 0
    fprintf('Rated %d for [movieId:%04d] %s\n', my_ratings(i), i, movieList{i});
  end
end
fprintf('\n\n');

pause;

% =============  =============  =============  =============  =============
% Learning Collaborative Filtering

% Setup data
Y = [my_ratings Y];
R = [(my_ratings~=0) R];

% Data normalization
[Ynorm Ymean] = normalizeRatings(Y, R);

[num_movies num_users] = size(Y);
num_features = 10;

% set initial parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% set Regularization
lambda = 10;

fprintf('Learning collaborative filtering...\n');
fflush(stdout);

% Optimization Theta and X using Collaborative Filtering
theta = fmincg(@(t)(cofiCostFunc(t, Ynorm, R, ...
                        num_users, num_movies, num_features, lambda)), ...
                initial_parameters, options);

% Unfold
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), num_users, num_features);

fprintf('Complete collaborative filtering...\n');
pause;


% =============  =============  =============  =============  =============
% Results

p = X * Theta';
my_predictions = p(:,1) + Ymean;

[r, ix] = sort(my_predictions, 'descend');

fprintf('\n\nTop recommendations for you:\n');
for i = 1:10
  j = ix(i);
  fprintf('Predicting rate %.1f for movie [movieId:%04d]%s\n', ...
            my_predictions(j), j, movieList{j});
end

