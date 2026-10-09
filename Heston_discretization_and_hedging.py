#some cool packages
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

#define function that will sample variance 
def variance_discretization(v_0, v_bar, gamma, kappa, T, n_paths, n_steps):
    dt=T/ n_steps
    v_t=np.zeros((n_paths, n_steps+1))
    v_t[:,0]=v_0

    #define all parameters not dependent on time step
    delta=4*kappa*v_bar/(gamma**2)
    c= gamma**2*(1-np.exp(-kappa*dt))/(4*kappa)
    nonc_factor=4*kappa*np.exp(-kappa*dt)/(gamma**2*(1-np.exp(-kappa*dt)))
    
    for i in range(n_steps):
        #noncentrality parameter
        nonc=nonc_factor*v_t[:,i]

        #sample from non-central chi-squared distribution
        v_t[:,i+1]= c*np.random.noncentral_chisquare(delta, nonc)

    return v_t

#define the function that will discretize the CIR process 
def Heston_discretization(S_0, r, v_0, v_bar, gamma, kappa, rho_xv, T, n_paths, n_steps):
    #first we calculata the varianceprocess
    dt=T/ n_steps
    v_t=np.zeros((n_paths, n_steps+1))
    v_t[:,0]=v_0
    
    #define all parameters not dependent on time step
    delta=4*kappa*v_bar/(gamma**2)
    c= gamma**2*(1-np.exp(-kappa*dt))/(4*kappa)
    nonc_factor=4*kappa*np.exp(-kappa*dt)/(gamma**2*(1-np.exp(-kappa*dt)))
        
    for i in range(n_steps):
        #noncentrality parameter
        nonc=nonc_factor*v_t[:,i]
    
        #sample from non-central chi-squared distribution
        v_t[:,i+1]= c*np.random.noncentral_chisquare(delta, nonc)


    #Now we can generate the correlated Brownian motions
    Z= np.random.normal(size=(n_paths, n_steps))

    #defien the parameters for the SDE
    k_0=(r-rho_xv*kappa*v_bar/gamma)*dt
    k_1=(rho_xv*kappa/gamma-0.5)*dt-rho_xv/gamma
    k_2=rho_xv/gamma
    k_3=(1-rho_xv**2)*dt

    #construct discretization fo SPDE
    x_t=np.zeros((n_paths, n_steps+1))
    x_t[:,0]=np.log(S_0)

    for i in range(n_steps):
        x_t[:,i+1]= x_t[:,i]+k_0+k_1*v_t[:,i]+k_2*v_t[:,i+1]+np.sqrt(k_3*v_t[:,i])*Z[:,i]

    #transform back to the original scale
    S_t=np.exp(x_t)

    return S_t, v_t


#define pricing and Delta function wrt to BS 
def BS_model(S_0, K, r, T, t_0, sigma, option_type):
    d1= (np.log(S_0/K)+(r+0.5*sigma**2)*(T-t_0))/(sigma*np.sqrt(T-t_0))
    d2= d1-sigma*np.sqrt(T-t_0)

    if option_type.lower()=='call':
        pricing=S_0 * norm.cdf(d1) - K * np.exp(-r * (T-t_0)) * norm.cdf(d2)
        delta=norm.cdf(d1)
        
        return pricing, delta
    
    elif option_type.lower()=='put':
        pricing=K * np.exp(-r * (T-t_0)) * norm.cdf(-d2) - S_0 * norm.cdf(-d1)
        delta=norm.cdf(d1)-1
        
        return pricing, delta
    
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

def Vega_BS(S_0, K, r, T, t_0, sigma):
    d1= (np.log(S_0/K)+(r+0.5*sigma**2)*(T-t_0))/(sigma*np.sqrt(T-t_0))
    vega=S_0 * norm.pdf(d1) * np.sqrt(T-t_0)
    
    return vega

#define pricing model wrt to Heston model, now not centered around X_0=ln(S_0/K)
#returns both the prciing and the Delta of the option
def COS_method_Heston(r, T, t_0, S_0, K, v_0, v_bar, gamma, kappa, rho_xv, option_type, L=10, N_k=500):
    """
    Prices European options under the Heston model using the COS method.
    Domain truncation [a, b] is calculated via sigma_std = sqrt(v_bar * tau).
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    tau = np.atleast_1d(np.asarray(T, dtype=float) - t_0)
    tau, K = np.broadcast_arrays(tau, K)
    
    #Truncation domain [a, b] centered around c_1
    X_0 = np.log(S_0 / K)
    c_1=r * tau + (1 - np.exp(-kappa * tau)) * (v_bar - v_0) / (2 * kappa) - 0.5 * v_bar * tau

    sigma_std = np.sqrt(v_bar * tau)
    
    a = c_1 - L * sigma_std
    b = c_1 + L * sigma_std
    
    # Fourier frequencies
    k = np.arange(N_k)
    k_grid = k[:, None]
    a_grid = a[None, :]
    b_grid = b[None, :]
    u_k = k_grid * np.pi / (b_grid - a_grid)
    
    # Heston Characteristic Function
    b_term = kappa - 1j * rho_xv * gamma * u_k
    D1 = np.sqrt(b_term**2 + (u_k**2 + 1j * u_k) * gamma**2)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        g = (b_term - D1) / (b_term + D1)
    # The zero frequency has phi(0)=1; the direct expression is 0/0 there.
    g[0] = 0.0
    
    exp_D1 = np.exp(-D1 * tau)
    D_minus = b_term - D1
    
    fact1 = np.exp(
        1j * u_k * r * tau + 
        (v_0 / gamma**2) * ((1 - exp_D1) / (1 - g * exp_D1)) * D_minus
    )
    fact2 = np.exp(
        (kappa * v_bar / gamma**2) * (
            tau * D_minus - 2 * np.log((1 - g * exp_D1) / (1 - g))
        )
    )
    phi_k = fact1 * fact2
    
    # Payoff Integrals (chi and psi)
    def chi(c, d):
        return (
            np.cos(k_grid * np.pi * (d - a_grid) / (b_grid - a_grid)) * np.exp(d)
            - np.cos(k_grid * np.pi * (c - a_grid) / (b_grid - a_grid)) * np.exp(c)
            + u_k * np.sin(k_grid * np.pi * (d - a_grid) / (b_grid - a_grid)) * np.exp(d)
            - u_k * np.sin(k_grid * np.pi * (c - a_grid) / (b_grid - a_grid)) * np.exp(c)
        ) / (1 + u_k**2)

    def psi(c, d):
        k_safe = np.where(k_grid == 0, 1.0, k_grid)
        val_nonzero = ((b_grid - a_grid) / (k_safe * np.pi)) * (
            np.sin(k_safe * np.pi * (d - a_grid) / (b_grid - a_grid)) - 
            np.sin(k_safe * np.pi * (c - a_grid) / (b_grid - a_grid))
        )
        return np.where(k_grid == 0, d - c, val_nonzero)

    # Normalized Payoff Coefficients U_k
    if option_type.lower() == 'call':
        H_k = K*(2/ (b_grid - a_grid)) * (chi(0, b_grid) - psi(0, b_grid))
    elif option_type.lower() == 'put':
        H_k = K*(2 / (b_grid - a_grid)) * (-chi(a_grid, 0) + psi(a_grid, 0))
    else:
        raise ValueError("option_type must be either 'call' or 'put'")
        
    #define summands for pricing and delta
    exp_k = np.exp(1j * k_grid * np.pi * (X_0[None, :] - a_grid) / (b_grid - a_grid))
    term = phi_k * H_k * exp_k


    #Summation for pricing and delta
    sum_val_pricing = np.sum(term, axis=0) - 0.5 * term[0]
    pricing =  np.exp(-r * tau) * sum_val_pricing.real

    sum_val_delta = np.sum(term*1j*u_k, axis=0) - 0.5 * term[0]*1j*u_k[0]
    delta =  np.exp(-r * tau) * sum_val_delta.real / S_0
    
    
    return pricing, delta


#define Vega function wrt to Heston model
def Vega_Heston(S_0, K, r, T, t_0, v_0, v_bar, gamma, kappa, rho_xv, option_type, L=10, N_k=500):
    K = np.atleast_1d(np.asarray(K, dtype=float))
    tau = np.atleast_1d(np.asarray(T, dtype=float) - t_0)
    tau, K = np.broadcast_arrays(tau, K)
        
    #Truncation domain [a, b] centered around c_1
    X_0 = np.log(S_0 / K)
    c_1=r * tau + (1 - np.exp(-kappa * tau)) * (v_bar - v_0) / (2 * kappa) - 0.5 * v_bar * tau

    sigma_std = np.sqrt(v_bar * tau)
        
    a = c_1 - L * sigma_std
    b = c_1 + L * sigma_std
        
    # Fourier frequencies
    k = np.arange(N_k)
    k_grid = k[:, None]
    a_grid = a[None, :]
    b_grid = b[None, :]
    u_k = k_grid * np.pi / (b_grid - a_grid)

    # Heston Characteristic Function
    b_term = kappa - 1j * rho_xv * gamma * u_k
    D1 = np.sqrt(b_term**2 + (u_k**2 + 1j * u_k) * gamma**2)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        g = (b_term - D1) / (b_term + D1)
    # The zero frequency has phi(0)=1; the direct expression is 0/0 there.
    g[0] = 0.0
        
    exp_D1 = np.exp(-D1 * tau)
    D_minus = b_term - D1
        
    fact1 = np.exp(
        1j * u_k * r * tau + 
        (v_0 / gamma**2) * ((1 - exp_D1) / (1 - g * exp_D1)) * D_minus
        )
    fact2 = np.exp(
        (kappa * v_bar / gamma**2) * (
            tau * D_minus - 2 * np.log((1 - g * exp_D1) / (1 - g))
            )
            )
    phi_k = fact1 * fact2

    #Now calculate the derivative of phi_k with respect to v
    c_k=(1-np.exp(-D1*tau))/(1-g*np.exp(-D1*tau))*D_minus/(gamma**2)
    d_phi_k=c_k*phi_k
        

    # Payoff Integrals (chi and psi)
    def chi(c, d):
        return (
            np.cos(k_grid * np.pi * (d - a_grid) / (b_grid - a_grid)) * np.exp(d)
            - np.cos(k_grid * np.pi * (c - a_grid) / (b_grid - a_grid)) * np.exp(c)
            + u_k * np.sin(k_grid * np.pi * (d - a_grid) / (b_grid - a_grid)) * np.exp(d)
            - u_k * np.sin(k_grid * np.pi * (c - a_grid) / (b_grid - a_grid)) * np.exp(c)
            ) / (1 + u_k**2)
    
    def psi(c, d):
        k_safe = np.where(k_grid == 0, 1.0, k_grid)
        val_nonzero = ((b_grid - a_grid) / (k_safe * np.pi)) * (
            np.sin(k_safe * np.pi * (d - a_grid) / (b_grid - a_grid)) - 
            np.sin(k_safe * np.pi * (c - a_grid) / (b_grid - a_grid))
            )
        return np.where(k_grid == 0, d - c, val_nonzero)
    
    # Payoff Coefficients H_k
    if option_type.lower() == 'call':
        H_k = (2*K / (b_grid - a_grid)) * (chi(0, b_grid) - psi(0, b_grid))
    elif option_type.lower() == 'put':
        H_k = (2*K / (b_grid - a_grid)) * (-chi(a_grid, 0) + psi(a_grid, 0))
    else:
        raise ValueError("option_type must be either 'call' or 'put'")


    # Summation
    exp_k= np.exp(1j * k_grid * np.pi * (X_0[None, :] - a_grid) / (b_grid - a_grid))
    term= d_phi_k * H_k * exp_k
    sum_val= np.sum(term, axis=0) - 0.5 * term[0]

    return np.exp(-r * tau) * sum_val.real

#define 


#We now look at a specific hedging strategy, where we are sell an option 
# and hedge it with the underlying asset. 
# We will calculate the PnL of this strategy under the Heston model.

#For these codes dt=(T-t_0)/n_hedges
#define PnL function with respect to the Heston model hedging 1 asset
def PnL_Heston(S_0, K, r, T, t_0, v_0, v_bar, gamma, kappa, rho_xv, option_type, n_paths, n_hedges, L=10, N_k=500):
    #We are given the amount of hedges and realizations we want to look at
    #Then we generate the stock price paths S with shape (n_paths, n_steps+1) 
    # and the corresponding time grid t with shape (n_steps+1,)
    S, v = Heston_discretization(S_0, r, v_0, v_bar, gamma, kappa, rho_xv, T, n_paths, n_hedges)
    t=np.linspace(t_0, T, n_hedges+1)
    dt= (T-t_0)/n_hedges



    #calculate the price and delta of the option at time t_0
    prices_t0, _ = COS_method_Heston(r, T, t_0, S_0, K, v_0, v_bar, gamma, kappa, rho_xv, option_type, L=L, N_k=N_k)
    prices_t0 = np.asarray(prices_t0).item()

    # The COS pricing routine is written for scalar inputs, so we evaluate delta path-by-path.
    deltas = np.zeros((n_paths, n_hedges))
    for p in range(n_paths):
        for j in range(n_hedges):
            _, delta_value = COS_method_Heston(
                r, T, t[j], S[p, j], K, v[p, j], v_bar, gamma, kappa, rho_xv, option_type, L=L, N_k=N_k
            )
            deltas[p, j] = np.asarray(delta_value).item()

    #define the PnL array
    PnL = np.zeros((n_paths, n_hedges+1))

    #PnL at time t_0
    PnL[:,0] = prices_t0 - deltas[:,0] * S_0

    #loop over the time steps to calculate PnL
    for i in range(n_hedges-1):
        #calculate the PnL at time t_i
        PnL[:,i+1] = PnL[:,i]*np.exp(r*(dt)) - (deltas[:,i+1]-deltas[:,i]) * S[:,i+1]

    #last step: calculate the PnL at maturity T
    if option_type.lower() == 'call':
        payoff = np.maximum(S[:,-1] - K, 0)
    elif option_type.lower() == 'put':
        payoff = np.maximum(K - S[:,-1], 0)

    
    #calculate the PnL at maturity
    PnL[:,-1] = PnL[:,-2]*np.exp(r*(dt)) - payoff + deltas[:,-1] * S[:,-1]
    
    return PnL, deltas, S, v


#For these codes dt=(T-t_0)/n_hedges
#calculate the PnL of a Hedging strategy under the Black-Scholes model with underlying having the Heston model
def PnL_BS(S_0, K, r, v_0, v_bar, gamma, kappa, rho_xv, T, t_0, sigma, option_type, n_paths, n_hedges):
    #We are given the amount of hedges and realizations we want to look at
    #Then we generate the stock price paths S with shape (n_paths, n_steps+1) 
    # and the corresponding time grid t with shape (n_steps+1,)
    S, v = Heston_discretization(S_0, r, v_0, v_bar, gamma, kappa, rho_xv, T, n_paths, n_hedges)
    t=np.linspace(t_0, T, n_hedges+1)
    dt= (T-t_0)/n_hedges
    
    #calculate the price and delta of the option at time t_0
    prices_t0, _ = BS_model(S_0, K, r, T, t_0, sigma, option_type)
    prices_t0 = np.asarray(prices_t0).item()

    # BS pricing is also scalar-based here, so we evaluate delta path-by-path.
    deltas = np.zeros((n_paths, n_hedges))
    for p in range(n_paths):
        for j in range(n_hedges):
            _, delta_value = BS_model(S[p, j], K, r, T, t[j], sigma, option_type)
            deltas[p, j] = np.asarray(delta_value).item()

    #define the PnL array
    PnL = np.zeros((n_paths, n_hedges+1))

    #PnL at time t_0
    PnL[:,0] = prices_t0 - deltas[:,0] * S_0

    #loop over the time steps to calculate PnL
    for i in range(n_hedges-1):
        #calculate the PnL at time t_i
        PnL[:,i+1] = PnL[:,i]*np.exp(r*(dt)) - (deltas[:,i+1]-deltas[:,i]) * S[:,i+1]

    #last step: calculate the PnL at maturity T
    if option_type.lower() == 'call':
        payoff = np.maximum(S[:,-1] - K, 0)
    elif option_type.lower() == 'put':
        payoff = np.maximum(K - S[:,-1], 0)

    
    #calculate the PnL at maturity
    PnL[:,-1] = PnL[:,-2]*np.exp(r*(dt)) - payoff + deltas[:,-1] * S[:,-1]
    
    return PnL, deltas, S, v





#Now we will hedge using 2 assets
def PnL_Heston_2assets(S_0, K, r, T, t_0, v_0, v_bar, gamma, kappa, rho_xv, option_type, n_paths, n_hedges, K_H, T_H, L=10, N_k=500):
    #We are given the amount of hedges and realizations we want to look at
    #Then we generate the stock price paths S with shape (n_paths, n_steps+1) 
    # and the corresponding time grid t with shape (n_steps+1,)
    S, v = Heston_discretization(S_0, r, v_0, v_bar, gamma, kappa, rho_xv, T, n_paths, n_hedges)
    t=np.linspace(t_0, T, n_hedges+1)
    dt= (T-t_0)/n_hedges



    #calculate the price, delta and Vega of the option we want to Hedge 
    prices_t0, _ = COS_method_Heston(r, T, t_0, S_0, K, v_0, v_bar, gamma, kappa, rho_xv, option_type, L=L, N_k=N_k)
    prices_t0 = np.asarray(prices_t0).item()

    # The COS pricing routine is written for scalar inputs, so we evaluate delta, vega path-by-path.
    #vegas is the vega of the derivative we want to hedge, vegas_h is the vega of the asset we use to hedge
    #V2 is the option used as hedging instrument
    deltas = np.zeros((n_paths, n_hedges))
    vegas= np.zeros((n_paths, n_hedges))
    vegas_V2=np.zeros((n_paths, n_hedges))
    V2=np.zeros((n_paths, n_hedges))
    deltas_V2=np.zeros((n_paths, n_hedges))
    V2_T=np.zeros(n_paths)


    for p in range(n_paths):
        v2_price_mat, _ = COS_method_Heston(r, T_H, T, S[p, -1], K_H, v[p, -1], v_bar, gamma, kappa, rho_xv, option_type, L=L, N_k=N_k)
        V2_T[p] = np.asarray(v2_price_mat).reshape(-1)[0]
        for j in range(n_hedges):
            _, delta_value = COS_method_Heston(
                r, T, t[j], S[p, j], K, v[p, j], v_bar, gamma, kappa, rho_xv, option_type, L=L, N_k=N_k
            )
            deltas[p, j] = np.asarray(delta_value).reshape(-1)[0]

            vega_value = Vega_Heston(S[p, j], K, r, T, t[j], v[p, j], v_bar, gamma, kappa, rho_xv, option_type, L=L, N_k=N_k)
            vegas[p, j] = np.asarray(vega_value).reshape(-1)[0]

            vega_V2_value = Vega_Heston(S[p, j], K_H, r, T_H, t[j], v[p, j], v_bar, gamma, kappa, rho_xv, option_type, L=L, N_k=N_k)
            vegas_V2[p, j] = np.asarray(vega_V2_value).reshape(-1)[0]

            V2_value, deltas_V2_value = COS_method_Heston(
                r, T_H, t[j], S[p, j], K_H, v[p, j], v_bar, gamma, kappa, rho_xv, option_type, L=L, N_k=N_k
            )

            V2[p, j] = np.asarray(V2_value).reshape(-1)[0]
            deltas_V2[p, j] = np.asarray(deltas_V2_value).reshape(-1)[0]

    

    #calculate the positions, 
    # phi_s is stock position and phi_V2 is hedging option position
    with np.errstate(divide='ignore', invalid='ignore'):
        phi_V2 = np.where(np.abs(vegas_V2) > 1e-12, vegas / vegas_V2, 0.0)
    phi_S = deltas - phi_V2 * deltas_V2
    
    
    #define the PnL array
    PnL = np.zeros((n_paths, n_hedges+1))

    #PnL at time t_0
    PnL[:,0] = prices_t0 - phi_S[:,0] * S_0-phi_V2[:,0]*V2[:,0]

    #loop over the time steps to calculate PnL
    for i in range(n_hedges-1):
        #calculate the PnL at time t_i
        PnL[:,i+1] = PnL[:,i]*np.exp(r*(dt)) - (phi_S[:,i+1]-phi_S[:,i]) * S[:,i+1]-(phi_V2[:,i+1]-phi_V2[:,i]) * V2[:,i+1]

    #last step: calculate the PnL at maturity T
    if option_type.lower() == 'call':
        payoff = np.maximum(S[:,-1] - K, 0)
    elif option_type.lower() == 'put':
        payoff = np.maximum(K - S[:,-1], 0)

    
    #calculate the PnL at maturity
    PnL[:,-1] = PnL[:,-2]*np.exp(r*(dt)) - payoff + phi_S[:,-1] * S[:,-1]+phi_V2[:,-1]*V2_T
    
    return PnL, phi_S, phi_V2, S


#Now we will do the same for the Black-Scholes model with underlying having the Heston model 
#2asset portfolio hedging
def PnL_BS_2assets(S_0, K, r, T, t_0, v_0, v_bar, gamma, kappa, rho_xv, sigma, option_type, n_paths, n_hedges, K_H, T_H):
    #We are given the amount of hedges and realizations we want to look at
    #Then we generate the stock price paths S with shape (n_paths, n_steps+1) 
    # and the corresponding time grid t with shape (n_steps+1,)
    S, v = Heston_discretization(S_0, r, v_0, v_bar, gamma, kappa, rho_xv, T, n_paths, n_hedges)
    t=np.linspace(t_0, T, n_hedges+1)
    dt= (T-t_0)/n_hedges



    #calculate the price, delta and Vega of the option we want to Hedge 
    prices_t0, _ = BS_model(S_0, K, r, T, t_0, sigma, option_type)
    prices_t0 = np.asarray(prices_t0).item()

    #vegas is the vega of the derivative we want to hedge, vegas_h is the vega of the asset we use to hedge
    #V2 is the option used as hedging instrument
    deltas = np.zeros((n_paths, n_hedges))
    vegas= np.zeros((n_paths, n_hedges))
    vegas_V2=np.zeros((n_paths, n_hedges))
    V2=np.zeros((n_paths, n_hedges))
    deltas_V2=np.zeros((n_paths, n_hedges))
    V2_T=np.zeros(n_paths)


    for p in range(n_paths):
        v2_price_mat, _ = BS_model(S[p, -1], K_H, r, T_H, T, sigma, option_type)
        V2_T[p] = np.asarray(v2_price_mat).reshape(-1)[0]
        for j in range(n_hedges):
            _, delta_value = BS_model(S[p, j], K, r, T, t[j], sigma, option_type)
            deltas[p, j] = np.asarray(delta_value).reshape(-1)[0]

            vega_value = Vega_BS(S[p, j], K, r, T, t[j], sigma)
            vegas[p, j] = np.asarray(vega_value).reshape(-1)[0]

            vega_V2_value = Vega_BS(S[p, j], K_H, r, T_H, t[j], sigma)
            vegas_V2[p, j] = np.asarray(vega_V2_value).reshape(-1)[0]

            V2_value, deltas_V2_value = BS_model(
                S[p, j], K_H, r, T_H, t[j], sigma, option_type
            )

            V2[p, j] = np.asarray(V2_value).reshape(-1)[0]
            deltas_V2[p, j] = np.asarray(deltas_V2_value).reshape(-1)[0]

    

    #calculate the positions, 
    # phi_s is stock position and phi_V2 is hedging option position
    with np.errstate(divide='ignore', invalid='ignore'):
        phi_V2 = np.where(np.abs(vegas_V2) > 1e-12, vegas / vegas_V2, 0.0)
    phi_S = deltas - phi_V2 * deltas_V2
    
    
    #define the PnL array
    PnL = np.zeros((n_paths, n_hedges+1))

    #PnL at time t_0
    PnL[:,0] = prices_t0 - phi_S[:,0] * S_0-phi_V2[:,0]*V2[:,0]

    #loop over the time steps to calculate PnL
    for i in range(n_hedges-1):
        #calculate the PnL at time t_i
        PnL[:,i+1] = PnL[:,i]*np.exp(r*(dt)) - (phi_S[:,i+1]-phi_S[:,i]) * S[:,i+1]-(phi_V2[:,i+1]-phi_V2[:,i]) * V2[:,i+1]

    #last step: calculate the PnL at maturity T
    if option_type.lower() == 'call':
        payoff = np.maximum(S[:,-1] - K, 0)
    elif option_type.lower() == 'put':
        payoff = np.maximum(K - S[:,-1], 0)

    
    #calculate the PnL at maturity
    PnL[:,-1] = PnL[:,-2]*np.exp(r*(dt)) - payoff + phi_S[:,-1] * S[:,-1]+phi_V2[:,-1]*V2_T
    
    return PnL, phi_S, phi_V2, S



def PnL_BS_COST(S_0, K, r, v_0, v_bar, gamma, kappa, rho_xv, T, t_0, sigma, cost, option_type, n_paths, n_hedges):
    #We are given the amount of hedges and realizations we want to look at
    #Then we generate the stock price paths S with shape (n_paths, n_steps+1) 
    # and the corresponding time grid t with shape (n_steps+1,)
    S, v = Heston_discretization(S_0, r, v_0, v_bar, gamma, kappa, rho_xv, T, n_paths, n_hedges)
    t=np.linspace(t_0, T, n_hedges+1)
    dt= (T-t_0)/n_hedges
    
    #calculate the price and delta of the option at time t_0
    prices_t0, _ = BS_model(S_0, K, r, T, t_0, sigma, option_type)
    prices_t0 = np.asarray(prices_t0).item()

    # BS pricing is also scalar-based here, so we evaluate delta path-by-path.
    deltas = np.zeros((n_paths, n_hedges))
    for p in range(n_paths):
        for j in range(n_hedges):
            _, delta_value = BS_model(S[p, j], K, r, T, t[j], sigma, option_type)
            deltas[p, j] = np.asarray(delta_value).item()

    #define the PnL array
    PnL = np.zeros((n_paths, n_hedges+1))
    PnL_cost = np.zeros((n_paths, n_hedges+1))

    #PnL at time t_0
    PnL[:,0] = prices_t0 - deltas[:,0] * S_0
    PnL_cost[:,0] = prices_t0 - deltas[:,0] * S_0

    #loop over the time steps to calculate PnL
    for i in range(n_hedges-1):
        #calculate the PnL at time t_i
        PnL[:,i+1] = PnL[:,i]*np.exp(r*(dt)) - (deltas[:,i+1]-deltas[:,i]) * S[:,i+1]
        PnL_cost[:,i+1] = PnL_cost[:,i]*np.exp(r*(dt)) - (deltas[:,i+1]-deltas[:,i]) * S[:,i+1] - cost * np.abs(deltas[:,i+1]-deltas[:,i]) * S[:,i+1]
    
    #last step: calculate the PnL at maturity T
    if option_type.lower() == 'call':
        payoff = np.maximum(S[:,-1] - K, 0)
    elif option_type.lower() == 'put':
        payoff = np.maximum(K - S[:,-1], 0)

    
    #calculate the PnL at maturity
    PnL[:,-1] = PnL[:,-2]*np.exp(r*(dt)) - payoff + deltas[:,-1] * S[:,-1]
    PnL_cost[:,-1] = PnL_cost[:,-2]*np.exp(r*(dt)) - payoff + deltas[:,-1] * S[:,-1]
    return PnL, PnL_cost, deltas, S, v




#Now we will plot the PnL of the hedging strategy under the Heston model and the Black-Scholes model with underlying having the Heston model
def PnL_summary(S_0, K, r, v_0, v_bar, gamma, kappa, rho_xv, T, t_0, sigma, option_type, n_paths, n_hedges, K_H, T_H, hist=False, path=False, stats=False):
    np.random.seed(123)
    PnL_Heston_model, deltas_Heston, S_Heston, _ = PnL_Heston(S_0, K, r, T, t_0, v_0, v_bar, gamma, kappa, rho_xv, option_type, n_paths, n_hedges)
    np.random.seed(123)
    PnL_BS_model, deltas_BS, S_BS, _ = PnL_BS(S_0, K, r, v_0, v_bar, gamma, kappa, rho_xv, T, t_0, sigma, option_type, n_paths, n_hedges)
    np.random.seed(123)
    PnL_Heston2, phi_S, phi_V2, S2= PnL_Heston_2assets(S_0, K, r, T, t_0, v_0, v_bar, gamma, kappa, rho_xv, option_type, n_paths, n_hedges, K_H, T_H)
    np.random.seed(123)
    PnL_BS2, phi_S_BS, phi_V2_BS, S2_BS= PnL_BS_2assets(S_0, K, r, T, t_0, v_0, v_bar, gamma, kappa, rho_xv, sigma, option_type, n_paths, n_hedges, K_H, T_H)
    
    #plot the histograms
    plt.hist(PnL_Heston_model[:,-1], bins=50, alpha=0.75, label='Heston Model PnL', color='navy', density=True)
    plt.hist(PnL_BS_model[:,-1], bins=50, alpha=0.7, label='BS Model PnL', color='darkorange', density=True)
    plt.title('PnL Distribution of single Asset Hedging Strategy')
    plt.xlabel('PnL')
    plt.ylabel('Frequency') 
    plt.legend()
    plt.show()
    plt.hist(PnL_Heston2[:,-1], bins=50, alpha=0.7, label='Heston Model PnL', color='darkgreen', density=True)
    plt.hist(PnL_BS2[:,-1], bins=50, alpha=0.7, label='BS Model PnL', color='purple', density=True)
    plt.title('PnL Distribution of 2 asset Hedging Strategy')
    plt.xlabel('PnL')
    plt.ylabel('Frequency') 
    plt.legend()
    plt.show()

    #calculate the mean and standard deviation of the PnL
    mean_PnL_Heston = np.mean(PnL_Heston_model[:,-1])
    std_PnL_Heston = np.std(PnL_Heston_model[:,-1])
    mean_PnL_BS = np.mean(PnL_BS_model[:,-1])
    std_PnL_BS = np.std(PnL_BS_model[:,-1])
    mean_PnL_Heston2 = np.mean(PnL_Heston2[:,-1])
    std_PnL_Heston2 = np.std(PnL_Heston2[:,-1])
    mean_PnL_BS2 = np.mean(PnL_BS2[:,-1])
    std_PnL_BS2 = np.std(PnL_BS2[:,-1])

    print(f'Mean PnL of Heston Model: {mean_PnL_Heston}', flush=True)
    print(f'Standard Deviation of PnL of Heston Model: {std_PnL_Heston}', flush=True)
    print(f'Mean PnL of BS Model: {mean_PnL_BS}', flush=True)
    print(f'Standard Deviation of PnL of BS Model: {std_PnL_BS}', flush=True)
    print(f'Mean PnL of 2 asset Heston Model: {mean_PnL_Heston2}', flush=True)
    print(f'Standard Deviation of PnL of 2 asset Heston Model: {std_PnL_Heston2}', flush=True)
    print(f'Mean PnL of 2 asset BS Model: {mean_PnL_BS2}', flush=True)
    print(f'Standard Deviation of PnL of 2 asset BS Model: {std_PnL_BS2}', flush=True)

    #only take the fisrt path for plotting
    PnL_Heston_model0 = PnL_Heston_model[0,:]
    deltas_Heston0 = deltas_Heston[0,:]
    S_Heston0 = S_Heston[0,:]
    PnL_BS_model0 = PnL_BS_model[0,:]
    deltas_BS0 = deltas_BS[0,:]
    S_BS0= S_BS[0,:]
    PnL_Heston2_path= PnL_Heston2[0,:]
    phi_S_path = phi_S[0,:]
    phi_V2_path = phi_V2[0,:]
    S2_path= S2[0,:]
    PnL_BS2_path= PnL_BS2[0,:]
    phi_S_BS_path = phi_S_BS[0,:]
    phi_V2_BS_path = phi_V2_BS[0,:]
    S2_BS_path= S2_BS[0,:]

    #plot the PnL path and comapre to the deltas and stock prices
    #for both models
    plt.plot(PnL_Heston_model0, color='darkorange', alpha=1.0, linewidth=0.8, label='Heston Model PnL')
    plt.plot(deltas_Heston0, color='purple', alpha=1.0, linewidth=0.8, label='Heston Model Deltas')
    plt.plot(S_Heston0, color='saddlebrown', alpha=1.0, linewidth=0.8, label='Heston Model Stock Prices')
    plt.title('PnL Path of single Asset Hedging Strategy under Heston Model')
    plt.xlabel('Time Steps')
    plt.ylabel('PnL')
    plt.legend()
    plt.grid(True, alpha=1)
    plt.show()
    plt.plot(PnL_BS_model0, color='darkorange', alpha=1.0, linewidth=0.8, label='BS Model PnL')
    plt.plot(deltas_BS0, color='purple', alpha=1.0, linewidth=0.8, label='BS Model Deltas')
    plt.plot(S_BS0, color='saddlebrown', alpha=1.0, linewidth=0.8, label='BS Model Stock Prices')
    plt.title('PnL Paths of single Asset Hedging Strategy under BS Model')
    plt.xlabel('Time Steps')
    plt.ylabel('PnL')
    plt.legend()
    plt.grid(True, alpha=1)
    plt.show()
    plt.plot(PnL_Heston2_path, color='darkorange', alpha=1.0, linewidth=0.8, label='Heston Model PnL')
    plt.plot(phi_S_path, color='purple', alpha=1.0, linewidth=0.8, label='Heston Model Stock pos')
    plt.plot(S2_path, color='saddlebrown', alpha=1.0, linewidth=0.8, label='Heston Model Stock Prices')
    plt.plot(phi_V2_path, color='black', alpha=1.0, linewidth=0.8, label='Heston Model risky asset pos')
    plt.title('PnL Path of 2 asset Hedging Strategy under Heston Model')
    plt.xlabel('Time Steps')
    plt.ylabel('PnL')
    plt.legend()
    plt.grid(True, alpha=1)
    plt.show()
    plt.plot(PnL_BS2_path, color='darkorange', alpha=1.0, linewidth=0.8, label='BS Model PnL')
    plt.plot(phi_S_BS_path, color='purple', alpha=1.0, linewidth=0.8, label='BS Model Stock pos')
    plt.plot(S2_BS_path, color='saddlebrown', alpha=1.0, linewidth=0.8, label='BS Model Stock Prices')
    plt.plot(phi_V2_BS_path, color='black', alpha=1.0, linewidth=0.8, label='BS Model risky asset pos')
    plt.title('PnL Path of 2 asset Hedging Strategy under BS Model')
    plt.xlabel('Time Steps')
    plt.ylabel('PnL')
    plt.legend()
    plt.grid(True, alpha=1)
    plt.show()

    #return the data
    return mean_PnL_Heston, std_PnL_Heston, mean_PnL_BS, std_PnL_BS, mean_PnL_Heston2, std_PnL_Heston2
    

def PnL_stats(S_0, K, r, v_0, v_bar, gamma, kappa, rho_xv, T, t_0, sigma, option_type, n_paths, n_hedges):
    np.random.seed(123)
    PnL_Heston_model, deltas_Heston, S_Heston, _ = PnL_Heston(S_0, K, r, T, t_0, v_0, v_bar, gamma, kappa, rho_xv, option_type, n_paths, n_hedges)
    np.random.seed(123)
    PnL_BS_model, deltas_BS, S_BS, _ = PnL_BS(S_0, K, r, v_0, v_bar, gamma, kappa, rho_xv, T, t_0, sigma, option_type, n_paths, n_hedges)

    #calculate the mean and standard deviation of the PnL
    mean_PnL_Heston = np.mean(PnL_Heston_model[:,-1])
    std_PnL_Heston = np.std(PnL_Heston_model[:,-1])
    mean_PnL_BS = np.mean(PnL_BS_model[:,-1])
    std_PnL_BS = np.std(PnL_BS_model[:,-1])

    print(f'Mean PnL of Heston Model: {mean_PnL_Heston}', flush=True)
    print(f'Standard Deviation of PnL of Heston Model: {std_PnL_Heston}', flush=True)
    print(f'Mean PnL of BS Model: {mean_PnL_BS}', flush=True)
    print(f'Standard Deviation of PnL of BS Model: {std_PnL_BS}', flush=True)



def PnL_path(S_0, K, r, v_0, v_bar, gamma, kappa, rho_xv, T, t_0, sigma, option_type, n_hedges, n_paths=2):
    np.random.seed(123)
    PnL_Heston_model, deltas_Heston, S_Heston, _ = PnL_Heston(S_0, K, r, T, t_0, v_0, v_bar, gamma, kappa, rho_xv, option_type, n_paths=n_paths, n_hedges=n_hedges)
    np.random.seed(123)
    PnL_BS_model, deltas_BS, S_BS, _ = PnL_BS(S_0, K, r, v_0, v_bar, gamma, kappa, rho_xv, T, t_0, sigma, option_type, n_paths=n_paths, n_hedges=n_hedges)

    #only take the fisrt path for plotting
    PnL_Heston_model = PnL_Heston_model[0,:]
    deltas_Heston = deltas_Heston[0,:]
    S_Heston = S_Heston[0,:]
    PnL_BS_model = PnL_BS_model[0,:]
    deltas_BS = deltas_BS[0,:]
    S_BS= S_BS[0,:]

    #plot the PnL path and comapre to the deltas and stock prices
    #for both models
    plt.plot(PnL_Heston_model, color='navy', alpha=1.0, linewidth=0.8, label='Heston Model PnL')
    plt.plot(deltas_Heston, color='darkgreen', alpha=1.0, linewidth=0.8, label='Heston Model Deltas')
    plt.plot(S_Heston, color='darkred', alpha=1.0, linewidth=0.8, label='Heston Model Stock Prices')
    plt.title('PnL Path of single Asset Hedging Strategy under Heston Model')
    plt.xlabel('Time Steps')
    plt.ylabel('PnL')
    plt.legend()
    plt.grid(True, alpha=1)
    plt.show()
    plt.plot(PnL_BS_model, color='darkorange', alpha=1.0, linewidth=0.8, label='BS Model PnL')
    plt.plot(deltas_BS, color='purple', alpha=1.0, linewidth=0.8, label='BS Model Deltas')
    plt.plot(S_BS, color='saddlebrown', alpha=1.0, linewidth=0.8, label='BS Model Stock Prices')
    plt.title('PnL Paths of single Asset Hedging Strategy under BS Model')
    plt.xlabel('Time Steps')
    plt.ylabel('PnL')
    plt.legend()
    plt.grid(True, alpha=1)
    plt.show()

    return deltas_Heston, deltas_BS



#Now we will define PnL cost comparaison
def PnL_cost_comparison(S_0, K, r, v_0, v_bar, gamma, kappa, rho_xv, T, t_0, sigma, option_type, n_paths):
    np.random.seed(123)
    PnL_BS_model_00, PnL_BS_cost_00, deltas_BS_00, S_BS_00, _ = PnL_BS_COST(S_0, K, r, v_0, v_bar, gamma, kappa, rho_xv, T, t_0, sigma, cost=0.0001, option_type=option_type, n_hedges=52, n_paths=n_paths)
    np.random.seed(123)
    PnL_BS_model_01, PnL_BS_cost_01, deltas_BS_01, S_BS_01, _ = PnL_BS_COST(S_0, K, r, v_0, v_bar, gamma, kappa, rho_xv, T, t_0, sigma, cost=0.0001, option_type=option_type, n_hedges=252, n_paths=n_paths)
    np.random.seed(123)
    PnL_BS_model_10, PnL_BS_cost_10, deltas_BS_10, S_BS_10, _ = PnL_BS_COST(S_0, K, r, v_0, v_bar, gamma, kappa, rho_xv, T, t_0, sigma, cost=0.005, option_type=option_type, n_hedges=52, n_paths=n_paths)
    np.random.seed(123)
    PnL_BS_model_11, PnL_BS_cost_11, deltas_BS_11, S_BS_11, _ = PnL_BS_COST(S_0, K, r, v_0, v_bar, gamma, kappa, rho_xv, T, t_0, sigma, cost=0.005, option_type=option_type, n_hedges=252, n_paths=n_paths)
    
    
    #plot the histograms, where we 
    plt.hist(PnL_BS_model_11[:,-1], bins=50, alpha=0.75, label='BS Model PnL', color='pink', density=True)
    plt.hist(PnL_BS_cost_11[:,-1], bins=50, alpha=0.7, label='BS Model PnL with Cost', color='darkorange', density=True)
    plt.title('PnL Distribution of single Asset Hedging Strategy with and without Cost, cost=0.005, n_hedges=252')
    plt.xlabel('PnL')
    plt.ylabel('Frequency') 
    plt.legend()
    plt.show()
    plt.hist(PnL_BS_cost_00[:,-1], bins=50, alpha=0.75, label='Low cost Low frequency', color='darkorange', density=True)
    plt.hist(PnL_BS_cost_01[:,-1], bins=50, alpha=0.7, label='Low cost High frequency', color='purple', density=True)
    plt.hist(PnL_BS_cost_10[:,-1], bins=50, alpha=0.7, label='High cost Low frequency', color='pink', density=True)
    plt.hist(PnL_BS_cost_11[:,-1], bins=50, alpha=0.7, label='High cost High frequency', color='red', density=True)
    plt.title('PnL Distribution of single Asset Hedging Strategy with and without Cost, cost=0.0001, n_hedges=252')
    plt.xlabel('PnL')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    return


#################################################
#Now we show the results
##################################################


#delta_Heston, delta_BS=PnL_path(S_0=1, K=1.1, r=0.05, v_0=0.04, v_bar=0.04, gamma=0.1, kappa=1, rho_xv=-0.7, T=1, t_0=0, sigma=0.2, option_type='call', n_hedges=252)


#print(delta_Heston[0:5],delta_BS[0:5])

PnL_summary(S_0=1, K=1.1, r=0.05, v_0=0.04, v_bar=0.04, gamma=0.1, kappa=0.1, rho_xv=-0.7, T=1, t_0=0, sigma=0.2, option_type='call', n_paths=1000, n_hedges=252, K_H=1.2, T_H=1.2, hist=True, path=True, stats=True)
PnL_cost_comparison(S_0=1, K=1.1, r=0.05, v_0=0.04, v_bar=0.04, gamma=0.1, kappa=0.1, rho_xv=-0.7, T=1, t_0=0, sigma=0.2, option_type='call', n_paths=1000)
    
print('Run Complete')