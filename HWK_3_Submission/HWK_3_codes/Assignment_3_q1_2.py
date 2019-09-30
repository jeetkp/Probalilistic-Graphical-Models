import numpy as np
from scipy.stats import norm
# from math import sin
import matplotlib.pyplot as plt

np.random.seed(2019);
samples= []
# Randon mean and covariance selected.
mu , sigma = 0,10

#Calculation for value of M
M=np.exp(1+(((np.pi)**2)/(2*(sigma**2)))) * np.sqrt(2*np.pi*(sigma**2))

#Counter
i=0
# Generate 10000 samples
while i<10000:
    q = np.random.normal(mu, sigma);
    u = np.random.uniform(0, 1);
    qdistri = norm.pdf(q,mu,sigma);

    #Range given from -pi to pi
    if np.absolute(q)>np.pi:
        # i=i+1;
        continue

    #Rejection or acception on the condition given
    if (u<=np.exp(np.sin(q))/(M*qdistri)):
        samples.append(q);
        i=i+1;

print(samples)
plt.hist(samples, bins='auto')
plt.show()
