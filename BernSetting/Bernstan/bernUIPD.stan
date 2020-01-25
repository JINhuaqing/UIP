functions {
    real fisherinfo(vector D){
        real thetahat;
        thetahat = mean(D);
        return 1.0/thetahat/(1.0-thetahat);
    }
}
data {
    int<lower=1> n;
    int<lower=1> n1;
    int<lower=1> n2;
    int D[n];
    vector[n1] D1;
    vector[n2] D2;
}
parameters {
    real<lower=1.0> M;
    simplex[2] pis;
    real<lower=0, upper=1> theta;
}
transformed parameters {
    real<lower=0, upper=1> mu;
    real<lower=0> sigma2;
    real alph;
    real bet;
    vector[2] gammas;

    mu = pis[1] * mean(D1) + pis[2] * mean(D2);
    sigma2 = 1/M/(pis[1] * fisherinfo(D1) + pis[2] * fisherinfo(D2));
    alph = mu * (mu*(1-mu)/sigma2 - 1);
    bet = (1-mu) * (mu*(1-mu)/sigma2 - 1);
    gammas = [1.0, 1.0]';
}

model {
    pis ~ dirichlet(gammas);
    M ~ uniform(2.0, n1+n2);
    theta ~ beta(alph, bet);
    for (i in 1:n)
        D[i] ~ bernoulli(theta);
}
