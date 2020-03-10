data {
    int<lower=1> n;
    int<lower=1> n1;
    int<lower=1> n2;
    int D[n];
    vector[n1] D1;
    vector[n2] D2;
}
parameters {
    real<lower=0, upper=1> theta;
    real<lower=0, upper=1> gamma1;
    real<lower=0, upper=1> gamma2;
}

transformed parameters{
  real<lower=0> alp;
  real<lower=0> bet;
  alp = 1 + gamma1 * sum(D1) + gamma2 * sum(D2);
  bet = 1 +  gamma1 * (n1 - sum(D1)) + gamma2 * (n2 -sum(D2));
}

model {
    gamma1 ~ beta(1, 1);
    gamma2 ~ beta(1, 1);
    theta ~ beta(alp, bet);
    for (i in 1:n)
        D[i] ~ bernoulli(theta);
}
