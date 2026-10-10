# Monte Carlo Simulation & Delta Hedging Engine

> **Work in Progress:** This repository is currently undergoing refactoring to optimize Monte Carlo performance and modularize simulation modules. Core simulation logic and hedging engine results are fully functional.

A computational framework for evaluating option hedging efficiency, tracking error, and PnL distributions under continuous-time stochastic dynamics.

## Features
- **Monte Carlo Engine:** Simulates asset price trajectories via Euler-Maruyama discretization under Black-Scholes and stochastic volatility models.
- **Dynamic Delta Hedging:** Simulates discrete-time rebalancing strategies to evaluate tracking error and hedging effectiveness.
- **PnL & Risk Analysis:** Quantifies residual PnL variance and hedging slippage under discrete rebalancing frequencies.

## Mathematical Framework
Asset trajectories are discretized over time step $dt$ using Euler-Maruyama:

$$S_{t+dt} = S_t + \mu S_t dt + \sigma S_t \sqrt{dt} \, Z_t, \quad Z_t \sim \mathcal{N}(0,1)$$

Dynamic Delta $\Delta(t) = \frac{\partial V}{\partial S}$ is rebalanced across discrete time intervals to monitor residual PnL Variance. For single asset hedging:

$$\text{PnL}(t_1)= V_0- \Delta_{t_0}S_{t_0}$$

$$\text{PnL}(t_{i+1})= \text{P nL}(t_{i})e^{r dt}- (\Delta(t_{i+1})-\Delta(t_{i}))S(t_{i+1})$$

$$\text{PnL}(T)= \text{P nL}(T-dt)e^{r dt}- \text{max}(S(T)-K,0)+\Delta(T-dt)S(T)$$


##Discussion Points
Discretizing the Heston model is a bit more difficult than the Black-Scholes model because of the CIR process, i.e., the variance process. The SPDE that governs its dynamics ensures nonnegativity, however this condition is revoked once we apply Euler-Maruyama. Therefore, we used conditional sampling, where $v(t)|v(s)$ follows a $\chi$-squared distribution that depends on $s,t$, and many other parameters. The correlation is then analytically injected in the discretization step via:
$$\sqrt{(k_3*v_t)}*Z$$

The histograms show that the BS model performs better in both the single-asset as well as the 2-asset hedging strategy. This can be seen by the high-narrow peaks of the BS histograms. This is surprising as one would assume that the hedging strategy that uses the same model as the underlying would perform better. 

Lastly, we introduced transaction costs into our model and compared weekly to daily hedging strategies. As expected, for high transaction costs weekly hedging performed the best and vice versa for low transaction costs. 


## Project Structure
```text
├── Figures                 # Results of the code
├── delta_hedging_sim.py    # Main Python script for Monte Carlo simulations & hedging
└── README.md               # Project overview and documentation
