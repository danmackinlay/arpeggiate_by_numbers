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

eps = np.finfo("float32").eps

def quantized_song_sparse(
        song,
        quantum=0.25,
        n_pitchclasses=12):
    song["step"] = np.rint(song.time/quantum)
    song["p"] = song.pitch % n_pitchclasses
    q_song = song.groupby(["step", "p"]).aggregate(dict(p=len))
    # q_song["time"] = q_song.index.get_level_values(0)*quantum
    q_song.rename(columns=dict(p="n"), inplace=True)
    return q_song

def quantized_song(
        song,
        quantum=0.25,
        n_pitchclasses=12):
    q_song_sp = quantized_song_sparse(
        song,
        quantum=quantum,
        n_pitchclasses=n_pitchclasses)
    n_steps = q_song_sp.index.get_level_values(0).max()+1
    all_pitchclasses = np.arange(n_pitchclasses, dtype=np.int32)
    all_steps = np.arange(n_steps, dtype=np.int32)
    #This could be done more efficiently with to_coo from pandas 0.16.0:
    dense_steps = [
        (step, p, q_song_sp.n.get((step,p),0))
        for step, p
        in itertools.product(all_steps, all_pitchclasses)]
    dense_frame=pd.DataFrame.from_records(dense_steps, index=[0,1],)
    dense_frame.rename_axis({2:"n"}, axis=1, inplace=True)
    dense_frame.index.rename(("step", "p",), inplace=True)
    return dense_frame

def gen_fields(obs,
        quantum=0.25,
        n_pitchclasses=12,
        frequencies=(0, 1.0/4, 1.0/3, 1.0/2, 2.0/3, 1.0, 4.0/3, 3.0/2, 2.0),
        half_lives=(1, 2, 4, 8, 16)
    ):
    """
    Exponential basis fields based on note occurrence
    """
    frequencies = np.asarray(frequencies)
    half_lives = np.asarray(half_lives)
    n_fields = 2 * frequencies.size * half_lives.size * n_pitchclasses
    ns = obs.shape[0]
    base_fields = np.zeros((ns, n_fields))
    base_field_pars = np.zeros((4, n_fields))
    return _gen_fields(
        obs, quantum, n_pitchclasses, frequencies, half_lives, ns,
        field_val_out=base_fields, field_par_out=base_field_pars)

def gen_fields_looped(obs,
        quantum=0.25,
        n_pitchclasses=12,
        frequencies=(0, 1.0/4, 1.0/3, 1.0/2, 2.0/3, 1.0, 4.0/3, 3.0/2, 2.0),
        half_lives=(1, 2, 4, 8, 16),
        loop_at=2.0,
    ):
    """
    Exponential basis fields based on note occurrence, looped to stationarity
    """
    frequencies = np.asarray(frequencies)
    half_lives = np.asarray(half_lives)
    n_fields = 2 * frequencies.size * half_lives.size * n_pitchclasses
    ns = obs.shape[0]
    step = int(loop_at/quantum)
    ns_looped = int(math.ceil(float(ns)/step)*step)
    base_fields_looped = np.zeros((ns_looped*2, n_fields))
    obs_looped = np.zeros((ns_looped*2, n_pitchclasses))
    obs_looped[:ns,:] = obs
    obs_looped[ns_looped:ns_looped+ns,:] = obs
    base_field_pars = np.zeros((4, n_fields))
    _gen_fields(
        obs_looped, quantum, n_pitchclasses, frequencies, half_lives,
        ns_looped*2,
        field_val_out=base_fields_looped, field_par_out=base_field_pars)
    return base_fields_looped[ns_looped:ns_looped+ns,:], base_field_pars


@numba.jit(nopython=True)
def _gen_fields(obs, quantum, n_pitchclasses, frequencies, half_lives, ns,
        field_val_out, field_par_out):
    twopi = math.pi *2
    n_f = frequencies.size
    n_h = half_lives.size
    for frequency_i in range(n_f):
        frequency = frequencies[frequency_i]
        
        for half_life_i in range(n_h):
            half_life = half_lives[half_life_i]

            for pitch in range(n_pitchclasses):
                field_i = (frequency_i * n_h * n_pitchclasses +
                           half_life_i * n_pitchclasses +
                           pitch) * 2
                #The frst two we interpret as del-p
                field_par_out[0, field_i] = pitch
                field_par_out[0, field_i+1] = pitch
                field_par_out[1, field_i] = 0 #phase
                field_par_out[1, field_i+1] = 1 #phase
                field_par_out[2, field_i] = frequency
                field_par_out[2, field_i+1] = frequency
                field_par_out[3, field_i] = half_life
                field_par_out[3, field_i+1] = half_life

            for s in range(ns):
                #print "t:", t, "decay:", decay, "osc:", math.cos(twopi*t/frequency), "frequency:", frequency, "half_life:", half_life
                t_s = s * quantum
                for r in range(s):
                    t_r = r * quantum
                    decay = 0.5**(t_r/half_life)
                    if decay < eps: continue
                    angle = twopi * t_r * frequency
                    cos_term = math.cos(angle) * decay * half_life
                    sin_term = math.sin(angle) * decay * half_life
                    for pitch in range(n_pitchclasses):
                        field_i = (frequency_i * n_h * n_pitchclasses +
                                   half_life_i * n_pitchclasses +
                                   pitch) * 2
                        field_val_out[s, field_i
                            ] += obs[s-r, pitch] * cos_term
                        field_val_out[s, field_i+1
                            ] += obs[s-r, pitch] * sin_term
    return field_val_out, field_par_out

def all_base_features(song,
        quantum=0.25,
        n_pitchclasses=12,
        frequencies=(0, 1.0/4, 1.0/3, 1.0/2, 2.0/3, 1.0, 4.0/3, 3.0/2, 2.0),
        half_lives=(1, 2, 4, 8, 16)
    ):
    """
    Return lists of features, one entry for each pitch class.
    Should I stack these?
    """
    #Response_values might be a more robust way of doing this?
    obs_a = song.n.values.reshape(-1, n_pitchclasses)
    fields, field_pars = gen_fields_looped(obs_a,
        quantum=quantum,
        n_pitchclasses=n_pitchclasses,
        frequencies=frequencies,
        half_lives=half_lives,
    )
    field_predictors = field_predictor_values(fields,
        n_pitchclasses=n_pitchclasses)
    return field_predictors, [c for c in obs_a.T], field_pars

def as_trials(field_predictor_list, obs_list, labels=None):
    """
    For now we only do the t-predictable regression.
    I insert an extra column of ones here to give us an offset for free
    """
    fp_nrows = field_predictor_list[0].shape[0]
    fp_ncols = field_predictor_list[0].shape[1]
    fp_nsamps = fp_nrows - 1
    n_params = fp_ncols + 1
    predictors = np.ones(
        ((len(field_predictor_list) * fp_nsamps),
        n_params)
    )
    #stack predictors into matrix with extra free col
    for i, field_predictor in enumerate(field_predictor_list):
        offset = i * fp_nsamps
        predictors[offset:offset+fp_nsamps, :-1] = field_predictor[:fp_nsamps,:]
    response = np.vstack([obs_vec.reshape(-1,1)[1:,:] for obs_vec in obs_list])
    return predictors, response


def rotator_for_pitch(n_pitchclasses=12, n_fields=1080, p=0):
    """
    a lookup index which will permute fields to be relative to the pitch under
    consideration.
    Hard to explain, easy to do.
    """
    n_pitch_phase_classes = n_pitchclasses *2
    predictor_rotate_part = (
        np.arange(n_pitch_phase_classes, dtype=np.int) + p*2
        ) % n_pitch_phase_classes
    predictor_rotate = (np.tile(
            predictor_rotate_part,
            n_fields/n_pitch_phase_classes
        ) + np.floor(
            np.arange(n_fields)/n_pitch_phase_classes) * n_pitch_phase_classes
    ).astype("int32")
    return predictor_rotate

def field_predictor_values_for_pitch(fields, n_pitchclasses=12, p=0):
    rotator = rotator_for_pitch(
        n_pitchclasses=n_pitchclasses,
        n_fields=fields.shape[1],
        p=p)
    return fields[:, rotator]

def field_predictor_values(fields, n_pitchclasses=12):
    # We don't need to do p=0 because it should be the same
    # Also this allows us to do a sanity cross-check
    return [fields] + [
        field_predictor_values_for_pitch(
            fields=fields,
            n_pitchclasses=n_pitchclasses,
            p=p)
        for p in range(1,n_pitchclasses)
    ]

def obs_values_for_pitch(song, n_pitchclasses=12, p=0):
    """obs values is unused for now"""
    return song.loc[song.index.get_level_values(1)==p
        ].values.reshape(-1, 1)

def obs_values(song, n_pitchclasses=12):
    """obs values is unused for now"""
    return [
        obs_values_for_pitch(
            song=song,
            n_pitchclasses=n_pitchclasses,
            p=p)
        for p in range(n_pitchclasses)
    ]
