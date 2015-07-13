from config import *
import numpy as np
import pandas as pd
import itertools
import numpy as np
import scipy as sp
import itertools
import random
import math
import numba
from scipy.optimize import minimize, approx_fprime, check_grad
from scipy.optimize.optimize import MemoizeJac
from numpy.linalg import norm
import pdb
from scipy.interpolate import InterpolatedUnivariateSpline, PchipInterpolator

eps_default = np.finfo("float32").eps

# lasso-Penalized estimate protocol for coordinate descent optimization
# we pass in the following arrays that will be updated.
# Penalties additive, corners only at zero

class PoissonLogScaler(object):
    """
    Log-linear regression object.
    """
    def fit(self, X, add_offset=True, eps=eps_default):
        # The parameters can have heavy tails and are exponentiated.
        # We always need to normalize to avoid unwished infinities
        self.add_offset = add_offset
        self.n_cols_in = X.shape[1]
        
        if self.add_offset:
            self.n_cols_out = self.n_cols_in + 1
        else:
            self.n_cols_out = self.n_cols_in
        self.X_loc = np.zeros(self.n_cols_out)
        self.X_scale = np.ones(self.n_cols_out)
        self.X_loc[:self.n_cols_in] = X.mean(0)
        self.X_scale[:self.n_cols_in] = np.maximum(X.std(0), eps)
        
    def transform(self, X):
        n_rows = X.shape[0]
        out = np.ones((n_rows, self.n_cols_out))
        out[:,:self.n_cols_in] = (X-self.X_loc[:self.n_cols_in]
            )/self.X_scale[:self.n_cols_in]
        return out
        
    def fit_transform(self, X, *args):
        self.fit(X, *args)
        return self.transform(X, *args)

    def untransform(X):
        #note this fine for data, but wrong for predictors
        return X * self.X_scale + self.X_loc
    
    def untransform_predictors():
        #TODO
        pass

def _pen_loss_alloc(predictors, response):
    """
    theta_hat, an estimate
    predictors (F-contiguous)
    pen_param size 1 penalty coefficient
    responses
    pen size 1 penalty array
    base_loss size 1 base loss val
    total_loss size 1 total loss array
    d_base_loss, base loss partial derivs
    dd_base_loss, base loss second partial derives
    d_pen, penalty partial derives
    dd_pen, penalty second partial derives
    d_total_loss, total loss partial derives
    dd_total_loss, total loss second partial derives
    pen_weight, penalty weights
    optional other state arrays
    """
    n_params = predictors.shape[1]
    n_trials = predictors.shape[0]
    theta_hat = np.zeros(n_params)
    pen_param = np.zeros(1)
    # predictors = np.asfortranarray(response) #col major
    predictors = np.ascontiguousarray(predictors) #row major
    response = np.reshape(response, -1)
    pen = np.zeros(1)
    base_loss = np.zeros(1)
    total_loss = np.zeros(1)
    d_base_loss = np.zeros(n_params)
    dd_base_loss = np.zeros(n_params)
    d_pen = np.zeros(n_params)
    dd_pen = np.zeros(n_params)
    d_total_loss = np.zeros(n_params)
    dd_total_loss = np.zeros(n_params)
    pen_weight = np.ones(n_params)
    log_lambdas = np.zeros(n_trials)
    lambdas = np.zeros(n_trials)
    return (
        theta_hat,
        pen_param,
        predictors,
        response,
        pen,
        base_loss,
        total_loss,
        d_base_loss,
        dd_base_loss,
        d_pen,
        dd_pen,
        d_total_loss,
        dd_total_loss,
        pen_weight,
        log_lambdas,
        lambdas,
    )

@numba.jit
def _loss_sum_l1_calc(
        theta_hat,
        pen_param,
        predictors,
        response,
        pen,
        base_loss,
        total_loss,
        d_base_loss,
        dd_base_loss,
        d_pen,
        dd_pen,
        d_total_loss,
        dd_total_loss,
        pen_weight,
        log_lambdas,
        lambdas,
    ):
    """summation fn for l1 losses
    NEARLY plain sum, but we handle 0 special"""
    total_loss[0] = base_loss[0] + pen[0]
    for i in range(theta_hat.size):
        d_total_loss[i] = ((d_base_loss[i] + d_pen[i]) * (
            (abs(d_pen[i])<abs(d_base_loss[i])) | (theta_hat[i] != 0.0)
        ))
        dd_total_loss[i] = dd_base_loss[i] + dd_pen[i]

@numba.jit
def _loglik_loss_exppoisson_calc(
        theta_hat,
        pen_param,
        predictors,
        response,
        pen,
        base_loss,
        total_loss,
        d_base_loss,
        dd_base_loss,
        d_pen,
        dd_pen,
        d_total_loss,
        dd_total_loss,
        pen_weight,
        log_lambdas,
        lambdas,
    ):
    #This could be optimized if we were only taking single coordinate derivatives
    base_loss[0] = 0
    d_base_loss[:] = 0
    dd_base_loss[:] = 0
    #first calculate lambdas, then losses
    for j in range(response.size):
        log_lambdas[j] = 0
        for i in range(theta_hat.size):
            log_lambdas[j] += predictors[j,i] * theta_hat[i]
        lambdas[j] = math.exp(log_lambdas[j])
        base_loss[0] -= response[j] * log_lambdas[j] - lambdas[j]
        for i in range(theta_hat.size):
            d_base_loss[i] -= predictors[j,i] * (response[j]-lambdas[j])
            dd_base_loss[i] += predictors[j,i] * predictors[j,i] * lambdas[j]


@numba.jit
def _loglik_loss_exppoisson_calc_i(
        i,
        val,
        theta_hat,
        pen_param,
        predictors,
        response,
        pen,
        base_loss,
        total_loss,
        d_base_loss,
        dd_base_loss,
        d_pen,
        dd_pen,
        d_total_loss,
        dd_total_loss,
        pen_weight,
        log_lambdas,
        lambdas,
    ):
    #untested, optimized for coordinate descent.
    # this will leave the other dimensions in an inconsistent state after update
    old_val = theta_hat[i]
    base_loss[0] = 0
    d_base_loss[i] = 0
    dd_base_loss[i] = 0
    #first calculate lambdas, then losses
    for j in range(response.size):
        log_lambdas[j] += predictors[j,i] * (val-old_val)
        lambdas[j] = math.exp(log_lambdas[j])
        base_loss[0] -= response[j] * log_lambdas[j] - lambdas[j]
        d_base_loss[i] -= predictors[j,i] * (response[j]-lambdas[j])
        dd_base_loss[i] += predictors[j,i] * predictors[j,i] * lambdas[j]

@numba.jit
def _penalty_l1_calc(
        theta_hat,
        pen_param,
        predictors,
        response,
        pen,
        base_loss,
        total_loss,
        d_base_loss,
        dd_base_loss,
        d_pen,
        dd_pen,
        d_total_loss,
        dd_total_loss,
        pen_weight,
        log_lambdas,
        lambdas,
    ):
    pen[0] = 0
    for i in range(theta_hat.size):
        pen[0] += abs(pen_weight[i] * theta_hat[i] * pen_param[0])
        d_pen[i] = math.copysign(pen_weight[i], theta_hat[i]) * pen_param[0]

class ExpPoissonL1Est(object):
    def __init__(self, predictors, response, pen_weight=None):
        self._allstate = _pen_loss_alloc(predictors, response)
        (
            self._theta_hat,
            self._pen_param,
            self._predictors,
            self._response,
            self._pen,
            self._base_loss,
            self._total_loss,
            self._d_base_loss,
            self._dd_base_loss,
            self._d_pen,
            self._dd_pen,
            self._d_total_loss,
            self._dd_total_loss,
            self._pen_weight,
            self._log_lambdas,
            self._lambdas,
        ) = self._allstate
        if pen_weight is None:
            self._pen_weight[:] = 1
            self._pen_weight[-1] = 0
        else:
            self._pen_weight[:] = 0
            self._pen_weight[:pen_weight.size] = pen_weights
    
    def cand_theta_hat(self):
        return np.zeros_like(self._theta_hat)

    def calc(self):
        self.base_loss_calc()
        self.penalty_calc()
        self.loss_sum_calc()

    def loss_sum_calc(self):
        _loss_sum_l1_calc(
            *self._allstate
        )

    def base_loss_calc(self):
        _loglik_loss_exppoisson_calc(
            *self._allstate
        )

    def penalty_calc(self):
        _penalty_l1_calc(
            *self._allstate
        )

    def set_theta_hat(self, theta_hat):
        ""
        self._theta_hat[:] = theta_hat

    def set_pen_param(self, pen_param):
        ""
        self._pen_param[:] = pen_param

    def pen_weight(self):
        return self._pen_weight[0]

    def total_loss(self):
        return self._total_loss[0]

    def base_loss(self):
        return self._base_loss[0]

    def pen(self):
        return self._pen[0]

    def theta_hat(self):
        return self._theta_hat

    def d_total_loss(self):
        return self._d_total_loss

    def d_base_loss(self):
        return self._d_base_loss

    def d_pen(self):
        return self._d_pen

    def dd_total_loss(self):
        """Note this does not return the full hessian, but the coordinate wise
        2nd partial derivs - the diagonal of the hessian"""
        return self._dd_total_loss

    def set_theta_hat_i(self, i, val):
        self._theta_hat[i] = val

    def theta_hat_i(self, i):
        return self._theta_hat[i]

    def d_total_loss_i(self, i=-1):
        "diff wrt i'th param"
        return self._d_total_loss[i]

    def dd_total_loss_i(self, i=-1):
        "2nd diff wrt ith param"
        return self._dd_total_loss[i]

    def d_base_loss_i(self, i=-1):
        "diff wrt i'th param"
        return self._d_base_loss[i]

    def dd_base_loss_i(self, i=-1):
        "diff wrt i'th param"
        return self._dd_base_loss[i]

    def d_pen_i(self, i=-1):
        "diff wrt i'th param"
        return self._d_pen[i]

    def dd_pen_i(self, i=-1, val=None):
        "diff wrt i'th param"
        return self._dd_pen[i]
    
    def lambdas(self):
        return self._lambdas
        
    def predictors(self):
        return self._predictors

    def response(self):
        return self._response

def coord_desc_reg_path(
        est,
        labels=None,
        n_penalties=None,
        update_iter=4,
        first_step_mul=2,
        stoch_coord=True,
        initial_noise=4.0,
        desc_rate_scale=1.0,
        almost_zero=eps_default,
        progress=True,
        debug=False,
        max_pen_param=None,
        pen_weights=None,
        norm_order=np.inf,
        update_offset=False
        ):
    """
    Calculate reg path for the field predictors and measured responses by
    coordinate descent.
    For now we assume log-linear regression.
    We will minimise penalized negative log likelihood.
    We will use a randomly initialised backwards algorithm so that we can do
    stability selection.
    """
    n_trials, n_params = est.predictors().shape
    if n_penalties is None:
        n_penalties = n_params * 2
    pen_params = np.zeros(n_penalties)
    logliks_base = np.zeros(n_penalties)
    logliks_total = np.zeros(n_penalties)
    n_nonzeros = np.zeros(n_penalties)
    theta_hat_sc_path = np.zeros((n_penalties, n_params))
    
    cand_theta_hat_sc = est.cand_theta_hat()
    #defaults - set the mean param as you'd expect
    resp_mean = est.response().mean()
    log_resp_mean = math.log(resp_mean)
    cand_theta_hat_sc[-1] = log_resp_mean
    # init:
    est.set_theta_hat(cand_theta_hat_sc)
    est.calc()
    null_penalty_map = np.abs(est.d_base_loss()) + almost_zero
    #these estimates are only approximate, but seem to give OK scaling
    if max_pen_param is not None:
        min_pen_param = np.min(null_penalty_map)
        pen_params[:] = min_pen_param * (max_pen_param/min_pen_param)**(np.arange(n_penalties)/(n_penalties-1))
    else:
        #We estimate likely increment values for domination by penalties
        rel_log_penalty_map = np.log(np.sort(
            null_penalty_map/(est.pen_weight()+almost_zero)))
        #OK, but we can have many identical, which is a giant waste of time computationally
        rel_log_penalty_increasing = np.concatenate(
            (
                ~np.isclose(
                    rel_log_penalty_map[1:],
                    rel_log_penalty_map[:-1]
                ),
                (True,)
            )
        )
        
        rel_log_penalty_map = rel_log_penalty_map[rel_log_penalty_increasing]
        rel_log_pen_slope = PchipInterpolator(
            np.linspace(0, 1, rel_log_penalty_map.size, endpoint=True),
            rel_log_penalty_map,
            extrapolate=True)
        pen_params[:] = np.exp(
            rel_log_pen_slope(np.linspace(0,1, n_penalties, endpoint=True))
        )
    # Choose random starting values by adding a small amount of noise
    # this noise is not perfect - biassed upward. 'twill serve.
    cand_theta_hat_sc[:-1] = np.random.normal(
        scale=1.0/math.sqrt(n_params),
        size=n_params-1) * initial_noise
    est.set_theta_hat(cand_theta_hat_sc)
    for pen_param_i in range(n_penalties):
        pen_param = pen_params[pen_param_i]
        est.set_pen_param(pen_param)
        est.penalty_calc()
        est.loss_sum_calc()
        
        if progress:
            print "============== pen_param", pen_param_i+1, "/", n_penalties, ":", pen_param, "=============="

        n_coord_updates = n_params * update_iter * (
            ((pen_param_i==0) * first_step_mul) + 1 )

        #select param index; could be uniformly at random;
        # or could use weighted selection, or plain order
        for coord_i_i in range(n_coord_updates):
            if stoch_coord:
                coord_i = np.random.randint(n_params)
            else:
                coord_i = coord_i_i % n_params
            if debug:
                print "======", coord_i_i, coord_i
            #Newton update based on derivatives
            d_i = est.d_total_loss_i(coord_i)
            if d_i==0.0:
                if debug:
                    print "null deriv, skipping."
                continue
                
            dd_i = est.dd_total_loss_i(coord_i)
            f = est.total_loss()
            curr_coord = est.theta_hat_i(coord_i)
            if debug:
                if curr_coord==0.0:
                    print "NULL0", est.d_total_loss_i(coord_i), "=", est.d_base_loss_i(coord_i), "+", est.d_pen_i(coord_i)
            
            coord_step_bound = 4
            step = -np.clip(d_i / dd_i, -coord_step_bound, coord_step_bound)
            base_step = step
            alpha = 1.0
            
            for propose_i in range(8):
                scaled_step = alpha*step
                if abs(scaled_step)<almost_zero:
                    if debug:
                        print "close enough dammit", cand_coord, scaled_step, d_i, dd_i
                        est.set_theta_hat_i(coord_i, curr_coord)
                        est.calc()
                    break
                
                cand_coord = curr_coord + scaled_step
                if (cand_coord>0 and curr_coord<0) or\
                        (cand_coord<0 and curr_coord>0):
                    cand_coord = 0.0
                est.set_theta_hat_i(coord_i, cand_coord)
                est.calc()
                cand_f = est.total_loss()
                cand_d_i = est.d_total_loss_i(coord_i)
                cand_dd_i = est.dd_total_loss_i(coord_i)
                if debug:
                    print propose_i, ")", f, curr_coord, d_i, alpha, alpha * step, "=>", cand_f, cand_coord, cand_d_i
                    print est.d_total_loss_i(coord_i), "=", est.d_base_loss_i(coord_i), "+", est.d_pen_i(coord_i)
                # if debug and pen_param_i>n_penalties/2 and propose_i>5:
                #     return debug_graph(est, curr_coord, base_step)
                if cand_f>f or not np.isfinite(cand_f):
                    # exploded. fall back to stepwise
                    alpha *= -0.5 #search even backwards?
                elif update_offset:
                    #success!
                    # also recalculate the mean by cheating
                    est.calc()
                    pred_mean = est.lambdas().mean()
                    curr_offset = est.theta_hat_i(-1)
                    mean_error = resp_mean/pred_mean
                    if debug:
                        print "updating mean", curr_offset, curr_offset + math.log(mean_error), "(", resp_mean, pred_mean, ")"
                    est.set_theta_hat_i(-1, curr_offset + math.log(
                        resp_mean/pred_mean))
                    est.calc()
                    break
                else:
                    break
            else:
                #put it back how we found it
                est.set_theta_hat_i(coord_i, curr_coord)
                est.calc()
                if debug:
                    print "not updating."
        theta_hat_sc_path[pen_param_i,:] = est.theta_hat()
        logliks_base[pen_param_i] = -est.base_loss()
        logliks_total[pen_param_i] = -est.total_loss()
        n_nonzero = (est.theta_hat()!=0).sum()
        n_nonzeros[pen_param_i] = n_nonzero
        if progress:
            print "===", n_nonzero, \
                norm(est.d_base_loss(), norm_order), \
                norm(est.theta_hat()*est.pen_weight(), norm_order)
        if n_nonzeros[pen_param_i] + n_nonzeros[max(pen_param_i-1,0)]<3:
            if debug:
                print "coordinates hit zero."
            break
    n_used_pen_params = pen_param_i + 1
    theta_hat_sc_path=theta_hat_sc_path[:n_used_pen_params,:]
    pen_params=pen_params[:n_used_pen_params]
    logliks_base=logliks_base[:n_used_pen_params]
    logliks_total=logliks_total[:n_used_pen_params]
    n_nonzeros=n_nonzeros[:n_used_pen_params]
    res = dict(
        theta_hat_sc_path=theta_hat_sc_path[:n_used_pen_params,:],
        pen_params=pen_params[:n_used_pen_params],
        logliks_base=logliks_base[:n_used_pen_params],
        logliks_total=logliks_total[:n_used_pen_params],
        n_nonzeros=n_nonzeros[:n_used_pen_params],
        aic=(n_nonzeros - logliks_base) * 2,
        aicc=((n_nonzeros - logliks_base) * 2
          + 2 * n_nonzeros * (n_nonzeros + 1)/(n_trials - n_nonzeros - 1))
    )
    return res

#for debugging
def test_grad_loglin(
        n_params=8,
        n_trials=23,
        mean=0.01, 
        n_nonzero=None,
        norm_order=1.0,
        randomize_out=True):
    if n_nonzero is None:
        n_nonzero = n_params - 1
    # These settings should mean we don't need rescaling
    logmean = math.log(mean)
    theta = np.zeros(n_params)
    theta[:n_nonzero] = np.random.normal(
        scale=1.0/math.sqrt(n_nonzero),
        size=(n_nonzero))
    theta[-1] = logmean
    predictors = np.random.normal(
        scale=1.0,
        size=(n_trials, n_params))
    predictors[:,-1] = 1
    lambdas = np.exp((predictors * theta.reshape(1,-1)).sum(1))
    if randomize_out:
        responses = np.random.poisson(lambdas, n_trials)
    else:
        responses = lambdas
    est = ExpPoissonL1Est(predictors, responses)
    def rand_vec(oracle_mean=False):
        v = np.random.normal(
            scale=1.0,
            size=n_params)
        if oracle_mean:
            v[-1] += logmean
        return v
    
    return dict(
        theta=theta,
        est=est,
        lambdas=lambdas,
        rand_vec=rand_vec,
    )

def debug_graph(est, curr_coord, base_step):
    print "WTF"
    thetas = np.linspace(curr_coord-abs(base_step), curr_coord+abs(base_step), 100)
    fs = np.zeros_like(thetas)
    d_ts = np.zeros_like(thetas)
    dd_ts = np.zeros_like(thetas)
    d_ps = np.zeros_like(thetas)
    d_ls = np.zeros_like(thetas)
    dd_ls = np.zeros_like(thetas)
    for i in range(100):
        theta = thetas[i]
        est.set_total_loss_i(coord_i, theta)
        est.calc()
        fs[i] = est.total_loss_i(coord_i)
        d_ts[i] = est.d_total_loss_i(coord_i)
        dd_ts[i] = est.dd_total_loss_i(coord_i)
        d_ps[i] = est.d_pen_i(coord_i)
        d_ls[i] = est.d_base_loss_i(coord_i)
        dd_ls[i] = est.dd_base_loss_i(coord_i)
    return dict(
        fs=fs,
        d_ts=d_ts,
        dd_ts=dd_ts,
        d_ps=d_ps,
        d_ls=d_ls,
        dd_ls=dd_ls,
        thetas=thetas,
        est=est,
        coord_i=coord_i,
        curr_coord=curr_coord,
        cand_coord=cand_coord,
    )
    