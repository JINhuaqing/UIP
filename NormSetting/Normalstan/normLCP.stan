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
    real logtau1;
    real logtau2;
}

transformed parameters{
  real mun;
  real<lower=0> tau1;
  real<lower=0> tau2;
  real<lower=0> sigma2n;
  real<lower=0> sigman;
  real<lower=0> sigma;
  real<lower=0> sigma2n1;
  real<lower=0> sigma2n2;
  real<lower=0> w1;
  real<lower=0> w2;

  tau1 = exp(logtau1);
  tau2 = exp(logtau2);
  sigma2n1 = sigma21 / n1 + 1.0/tau1;
  sigma2n2 = sigma22 / n2 + 1.0/tau2;
  w1 = 1/sigma2n1;
  w2 = 1/sigma2n2;
  mun = (mD1 * w1 + mD2 * w2)/(w1 + w2);
  sigma2n = 1/(w1 + w2);

  sigman = sigma2n^(0.5);
  sigma = sigma2^(0.5);
}

model {
    logtau1 ~ uniform(-30, 30);
    logtau2 ~ uniform(-30, 30);
    sigma2 ~ inv_gamma(0.01, 0.01);
    theta ~ normal(mun, sigman);
    for (i in 1:n)
        D[i] ~ normal(theta, sigma);
}
