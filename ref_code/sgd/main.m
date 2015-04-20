clear;
load;

% Parameters
n_users = max(data(:,1));
n_movies = max(data(:,2));
K = 20;
n_iterations = 20;

M = size(data, 1);

lambda = 0.01 ;
learning_rate = 0.005;
% since the output of the function is gradient / lambda
% our rate is learning_rate * lambda
rate = learning_rate;
mu = mean(data(:,4));

% Randomly Initialize all the Xs
U = random('unif', 0, 1, [K, n_users]);
V = random('unif', 0, 1, [K, n_movies]);
A = random('unif', 0, 1, [1, n_users]);
B = random('unif', 0, 1, [1, n_movies]);

Y = data(:,4);
I = data(:,1);
J = data(:,2);

shrink = 1 - rate * lambda

for i_loop = 1:n_iterations
    fprintf('start iter %d', i_loop);
    perm = randperm(M);
    tic;
    for i_perm = 1:M
        index = perm(i_perm);
        i = I(index, 1);
        j = J(index, 1);
        y = Y(index, 1);
        [gUi, gVj, gAi, gBj] = sgd_gradient(y, U, V, A, B, lambda, mu, i, j);
        
        U(:,i) = U(:,i) - rate * gUi;
        V(:,j) = V(:,j) - rate * gVj;
        A(:,i) = A(:,i) - rate * gAi;
        B(:,j) = B(:,j) - rate * gBj;
        
        if mod(i_perm, 10000) == 0
            toc;tic;
            i_perm
        end 
    end
    U = shrink * U;
    V = shrink * V;
    A = shrink * A;
    B = shrink * B;
    
    
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

