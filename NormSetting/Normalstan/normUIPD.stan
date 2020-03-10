data {
    int<lower=1> n;
    int<lower=1> n1;
    int<lower=1> n2;
    vector[n] D;
    vector[n1] D1;
    vector[n2] D2;
}
transformed data {
    real<lower=0> sigma21;
    real<lower=0> sigma22;
    real mD1;
    real mD2;
    sigma21 = variance(D1);
    sigma22 = variance(D2);
    mD1 = mean(D1);
    mD2 = mean(D2);

}
parameters {
    real theta;
    real<lower=0> sigma2; 
    real<lower=1> M;
    simplex[2] pis;
}

transformed parameters{
  real mun;
  real<lower=0> sigma2n;
  real<lower=0> sigman;
  real<lower=0> sigma;
  mun = pis[1] * mD1 + pis[2] * mD2;
  sigma2n = 1.0 /M/(pis[1]/sigma21 + pis[2]/sigma22);
  sigman = sigma2n ^ (0.5);
  sigma = sigma2 ^ (0.5);
}

model {
    pis ~ dirichlet([1.0, 1.0]');
    M ~ uniform(0, n1+n2);
    sigma2 ~ inv_gamma(0.01, 0.01);
    theta ~ normal(mun, sigman);
    for (i in 1:n)
        D[i] ~ normal(theta, sigma);
}
