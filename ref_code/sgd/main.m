clear;
load;

% Parameters
n_users = max(data(:,1));
n_movies = max(data(:,2));
K = 5;
n_iterations = 20;

M = size(data, 1);

lambda = 0.01 ;
learning_rate = 0.005;


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
    perm_i = randperm(ceil(M / 1e5));
    tic;
    for i_perm = perm_i
        perm_j = randperm(1e5);
        for j_perm = perm_j
            index = ((i_perm) - 1) * 1e4 + j_perm;
            if(index <= M)                
                i = I(index, 1);
                j = J(index, 1);
                y = Y(index, 1);
                [gUi, gVj, gAi, gBj] = sgd_gradient(y, U, V, A, B, lambda, mu, i, j);

                U(:,i) = U(:,i) - rate * gUi;
                V(:,j) = V(:,j) - rate * gVj;
                A(:,i) = A(:,i) - rate * gAi;
                B(:,j) = B(:,j) - rate * gBj;
            end
        end
        toc;tic;
        i_perm * 1e5
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

