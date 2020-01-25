functions {
    real KLnorm(real mu1, real mu2, real sigma1, real sigma2){
        real itm1;
        real itm2;
        itm1 = log(sigma2/sigma1);
        itm2 = (sigma1^2 + (mu2 - mu1)^2)/(2*sigma2^2) - 0.5;
        return itm1 + itm2;
    }

    real JSnorm(real mu1, real mu2, real sigma1, real sigma2){
         real KL1;
         real KL2;
         KL1 = KLnorm(mu1, mu2, sigma1, sigma2);
         KL2 = KLnorm(mu2, mu1, sigma2, sigma1);
         return (KL1 + KL2)/2;
    }
}
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
    real<lower=0> sigma2hat;
    vector<lower=0>[3] sigma2inits;
    vector[3] muinits;
    vector[2] InvSJs;
    simplex[2] pis;
    real mun;

    real mD1;
    real mD2;
    real sD1;
    real sD2;
    real sD;
    mD1 = mean(D1);
    mD2 = mean(D2);
    sigma21 = variance(D1);
    sigma22 = variance(D2);
    sigma2hat = variance(D);
    sD1 = sum(D1);
    sD2 = sum(D2);
    sD = sum(D);
    muinits[1] = sD1/(n1+sigma21/100.0);
    muinits[2] = sD2/(n2+sigma22/100.0);
    muinits[3] = sD/(n+sigma2hat/100.0);
    sigma2inits[1] = 1 / (1/100.0 + n1/sigma21);
    sigma2inits[2] = 1 / (1/100.0 + n2/sigma22);
    sigma2inits[3] = 1 / (1/100.0 + n/sigma2hat);
    InvSJs[1] = 1.0 / (JSnorm(muinits[1], muinits[3], sigma2inits[1]^(0.5), sigma2inits[3]^(0.5)) + 1e-10);
    InvSJs[2] = 1.0 / (JSnorm(muinits[2], muinits[3], sigma2inits[2]^(0.5), sigma2inits[3]^(0.5)) + 1e-10);
    pis[1] = InvSJs[1] / sum(InvSJs);
    pis[2] = InvSJs[2] / sum(InvSJs);
    mun = pis[1] * mD1 + pis[2] * mD2;
}

parameters {
    real theta;
    real<lower=0> sigma2; 
    real<lower=1> M;
}

transformed parameters{
  real<lower=0> sigma2n;
  real<lower=0> sigman;
  real<lower=0> sigma;
  sigma2n = 1.0 /M/(pis[1]/sigma21 + pis[2]/sigma22);
  sigman = sigma2n ^ (0.5);
  sigma = sigma2 ^ (0.5);
}

model {
    M ~ uniform(2.0, n1+n2);
    sigma2 ~ inv_gamma(0.01, 0.01);
    theta ~ normal(mun, sigman);
    for (i in 1:n)
        D[i] ~ normal(theta, sigma);
}
