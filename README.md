## This repo is the simulation code for Unit Information Prior (UIP)


### Simulation Settings

- Binary data
   - The sampling of UIP methods are based on both MCMC sampling and rejection sampling.
   - The sampling of MPP is based on stan.
   
- Continuous data
   - The sampling of MPP, LCP and UIP methods is based on PyMC3 package in python.
   - The sampling of Jeffreys' prior is based on the Gibbs sampler.

### Real data 

The real data is about six clinical trials of Alzheimer's disease (AD).

And we apply UIP-Dirichlet on these datasets.
