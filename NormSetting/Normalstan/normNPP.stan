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
    real sD1;
    real sD2;
    sigma21 = variance(D1);
    sigma22 = variance(D2);
    sD1 = sum(D1);
    sD2 = sum(D2);

}
parameters {
    real theta;
    real<lower=0> sigma2; 
    real<lower=0, upper=1> gamma1;
    real<lower=0, upper=1> gamma2;
}

transformed parameters{
  real mun;
  real<lower=0> sigma2n;
  real<lower=0> sigman;
  real<lower=0> sigma;
  mun = (gamma1 * sD1/sigma21 +  gamma2 * sD2/sigma22)/(gamma1*n1/sigma21 + gamma2*n2/sigma22);
  sigma2n = 1.0 /(gamma1*n1/sigma21 + gamma2*n2/sigma22); 
  sigman = sigma2n ^ (0.5);
  sigma = sigma2 ^ (0.5);
}

model {
    gamma1 ~ beta(1.0, 1.0);
    gamma2 ~ beta(1.0, 1.0);
    sigma2 ~ inv_gamma(0.01, 0.01);
    theta ~ normal(mun, sigman);
    for (i in 1:n)
        D[i] ~ normal(theta, sigma);
}
