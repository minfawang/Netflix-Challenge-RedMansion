data = load('data.txt');

% Parameters
n_users = 943;
n_movies = 1682;
K = 20;
n_iterations = 100;

M = size(data, 1);

lambda = 0.01 / (n_user * n_movies);
learning_rate = 0.005;
% since the output of the function is gradient / lambda
% our rate is learning_rate * lambda
rate = lambda * learning_rate;
mu = mean(data(:,3));

% Randomly Initialize all the Xs
U = random('unif', 0, 1, [K, n_users]);
V = random('unif', 0, 1, [K, n_movies]);
A = random('unif', 0, 1, [1, n_users]);
B = random('unif', 0, 1, [1, n_movies]);

Y = data(:,3);
I = data(:,1);
J = data(:,2);

for i_loop = 1:n_iterations
    perm = randperm(M);
    for i_perm = 1:M
        index = perm(i_perm);
        i = I(index, 1);
        j = J(index, 1);
        y = Y(index, 1);
        [gU, gV, gA, gB] = sgd_gradient(y, U, V, A, B, lambda, mu, i, j);
        
        U = U - rate * gU;
        V = V - rate * gV;
        A = A - rate * gA;
        B = B - rate * gB;
    end
    
    Y_guess = zeros(M, 1);
    for i_row = 1:M
        i = I(i_row, 1);
        j = J(i_row, 1);
        Y_guess(i_row, 1) = U(:,i)' * V(:,j) + A(:,i) + B(:,j) + mu;
    end
    
    err = Y - Y_guess;
    err2 = mean(err.^2);
    err2
end

% Save to 'matlab.mat'
save;
