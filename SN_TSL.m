function [Q, W, Omega, obj_value] = SN_TSL(X, c,  H, alpha, beta, lambda) 

tol = 1e-6; 
rho = 1.1;
max_mu = 1e8;
mu = 1e-5;
maxIter = 20;
[d, n] = size(X);

%% initialize 
r = 2 * c;
W = normcol_equal(randn(r, d));
Q = normcol_equal(randn(c, r));

J = zeros(r, n);
P = zeros(d, d);
Y1 = zeros(r, n);
Wtemp = inv(X * X' + lambda * eye(d)); 

%% starting iterations
iter = 0;
while iter < maxIter
    iter = iter + 1; 
    
    %update Omega
    
    Omega = inv(beta * Q' * Q + (mu + 1) * eye(r)) * (W * X + beta * Q' * H + mu * J - Y1);
    Omega = max(Omega, 0);
    
    
    %update J
    
    Jtemp = Omega + mu \ Y1;
    J = max(0, Jtemp - mu \ alpha) + min(0, Jtemp + mu \ alpha);
    
    

    %update W 
    
    W = Omega * X' * Wtemp;
    


    %update Q   
    
    Q = beta * H * Omega' * inv(beta * Omega * Omega' + lambda * eye(r));
    


%     obj_value(iter) = 2 \ norm(W * X - Omega, 'fro') ^ 2 + alpha * sum(sum(abs(Omega))) + 2 \ beta * norm(Q * Omega - H, 'fro') ^ 2 + 2 \ lambda * (norm(W, 'fro') ^ 2 + norm(Q, 'fro') ^ 2);
   obj_value = 0;
   
   %% convergence check   
   
    leq1 = Omega - J;


    stopC = max(max(abs(leq1)));
%     stopC = max(stopC, max(max(abs(leq2))));

    if stopC < tol || iter >= maxIter
        break;
    else
        Y1 = Y1 + mu * leq1;
%         Y2 = Y2 + mu * leq2;
%         Y3 = Y3 + mu * leq3;
        mu = min(max_mu, mu * rho);
    end
    
%     if (iter==1 || mod(iter, 5 )==0 || stopC<tol)
%             disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
%             ',stopALM=' num2str(stopC,'%2.3e') ]);
%     end

end

end

% Solving L_{21} norm minimization
function [E] = solve_l1l2(W,lambda)
n = size(W,1);
E = W;
for i=1:n
    E(i,:) = solve_l2(W(i,:),lambda);
end
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
end

