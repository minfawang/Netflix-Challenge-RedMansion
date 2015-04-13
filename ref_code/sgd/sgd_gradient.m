function [ gU, gV, gA, gB ] = sgd_gradient( y, U, V, A, B, lambda, mu, i, j)
    % for efficience, output gradient / lambda
    gU = U;
    gV = V;
    gA = A;
    gB = B;
    pFpX = 2 * (y - mu - (U(:,i)' * V(:,j) + A(:,i) + B(:,j))) / lambda;
    gU(:,i) = gU(:,i) - pFpX * V(:,j);
    gV(:,j) = gV(:,j) - pFpX * U(:,i);
    gA(:,i) = gA(:,i) - pFpX;
    gB(:,j) = gB(:,j) - pFpX;
end

