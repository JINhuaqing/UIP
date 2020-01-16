data {
    int<lower=1> n;
    int<lower=1> n1;
    int<lower=1> n2;
    int D[n];
    vector[n1] D1;
    vector[n2] D2;
}
transformed data {
    real<lower=0> sigma21;
    real<lower=0> sigma22;
    sigma21 = variance(D1);
    sigma22 = variance(D2);

}
parameters {
    real<lower=0, upper=1> theta;
    real<lower=0> sigma2; 
    real<lower=0, upper=1> gamma1;
    real<lower=0, upper=1> gamma2;
}

transformed parameters{
  real mun;
  real<lower=0> sigma2n;
  real<lower=0> sigman;
  real<lower=0> sigma;
  mun = (gamma1 * sum(D1)/sigma21 +  gamma2 * sum(D2)/sigma22)/(gamma1*n1/sigma21 + gamma2*n2/sigma22);
  sigma2n = 1.0 /(gamma1*n1/sigma21 + gamma2*n2/sigma22); 
  sigman = sigma2n ^ (1/2);
  sigma = sigma ^ (1/2);
}

model {
    gamma1 ~ beta(1, 1);
    gamma2 ~ beta(1, 1);
    sigma2 ~ inv_gamma(0.01, 0.01);
    theta ~ beta(mun, sigman);
    for (i in 1:n)
        D[i] ~ normal(theta, sigma);
}
