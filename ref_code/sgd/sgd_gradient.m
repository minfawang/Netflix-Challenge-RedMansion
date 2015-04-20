function [ gUi, gVj, gAi, gBj ] = sgd_gradient( y, U, V, A, B, lambda, mu, i, j)
    % for efficience, output gradient / lambda
    pFpX = 2 * (y - mu - (U(:,i)' * V(:,j) + A(:,i) + B(:,j)));

    gUi = - pFpX * V(:,j);
    gVj = - pFpX * U(:,i);
    gAi =  - pFpX;
    gBj =  - pFpX;
end

