import math
import numpy as np
import pandas as pd
from typing import Dict
from config import Config
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.') 

def argss():
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Args for Env""")
    prs.add_argument("-a", dest="lr", type=float,default=0.0003, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-gamma", dest="gamma", type=float,default=0.99, required=False, help="discount factor.\n")
    prs.add_argument("-ep", dest="n_epochs_per_update", type=int, default=10, required=False, help="acceleration of t.\n")
    prs.add_argument("-batch", dest="batch_size", type=int, default=128, required=False, help="batch_size. \n")
    prs.add_argument("-buffer", dest="buffer", type=int, default=2048, required=False, help="batch_size. \n")
    prs.add_argument("-ls", dest="learning_starts", type=int, default=100, required=False, help="batch_size. \n")
    prs.add_argument("-cr", dest="clip_range", type=float, default=0.2, required=False, help="clip_range. \n")
    prs.add_argument("-vfc", dest="vf_coef", type=float, default=0.5, required=False, help="vf coef. \n")
    prs.add_argument("-efc", dest="ent_coef", type=float, default=0.0, required=False, help="ent coef. \n")
    prs.add_argument("-actnorm", dest="act_norm", type=str2bool, default=True, required=False, help="Whether to normalize actions.\n")
    prs.add_argument("-statenorm", dest="state_norm", type=str2bool, default=True, required=False, help="Whether to normalize states.\n")
    prs.add_argument("-data", dest="data", type=str, default="cpu", required=False, help="Data type: cpu or gpu.\n")
    prs.add_argument('-reward', dest='reward_type', type=str, default="cpu", required=False, help="Reward function: cpu or gpu.\n")
    
    args = prs.parse_args()
    return args

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.std = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch):
        batch = np.array(batch)
        bmean = np.mean(batch, axis=0)
        bvar = np.var(batch, axis=0)
        bcount = batch.shape[0]
        delta = bmean - self.mean
        tot = self.count + bcount
        self.mean = self.mean + delta * bcount / tot
        m_a = self.var * self.count
        m_b = bvar * bcount
        M2 = m_a + m_b + (delta**2) * self.count * bcount / tot
        self.var = M2 / tot
        self.std = np.sqrt(self.var + 1e-8)
        self.count = tot

# class RunningMeanStd:
#     def __init__(self, epsilon=1e-4):
#         self.mean = 0
#         self.var = 1
#         self.std = 1
#         self.count = epsilon

#     def update(self, batch):
#         bmean = np.mean(batch)
#         bvar = np.var(batch)
#         bcount = len(batch)
#         delta = bmean - self.mean
#         tot = self.count + bcount
#         self.mean = self.mean + delta * bcount / tot
#         m_a = self.var * self.count
#         m_b = bvar * bcount
#         M2 = m_a + m_b + (delta**2) * self.count * bcount / tot
#         self.var = M2 / tot
#         self.std = np.sqrt(self.var + 1e-8)
#         self.count = tot


def simulate_twap_taker(processed_df: pd.DataFrame, config: Config) -> Dict:
    """
    Simple TWAP: split inventory evenly across horizon, execute as taker at L1 each step.
    Uses L1..L3 (if L1 insufficient) for realistic VWAP of fills.
    """
    remain = config.initial_inventory
    qty_per_step = math.ceil(config.initial_inventory / config.time_horizon)
    executed_qty, executed_value = 0, 0.0
    vwap_num, vwap_den = 0.0, 0.0
    trades = 0

    for t in range(min(config.time_horizon, len(processed_df))):
        if remain <= 0: break
        row = processed_df.iloc[t]
        this_qty = min(qty_per_step, remain)
        if this_qty <= 0: continue

        # taker fill
        filled, cash = 0, 0.0
        if config.side == 'buy':
            for px, vol in [(row['askPrice1'], row['askVolume1']),
                            (row['askPrice2'], row['askVolume2']),
                            (row['askPrice3'], row['askVolume3']),
                            (row['askPrice4'], row['askVolume4']),
                            (row['askPrice5'], row['askVolume5'])]:
                take = int(min(this_qty - filled, max(0, vol)))
                if take <= 0: continue
                filled += take; cash += take*px
                if filled >= this_qty: break
        else:
            for px, vol in [(row['bidPrice1'], row['bidVolume1']),
                            (row['bidPrice2'], row['bidVolume2']),
                            (row['bidPrice3'], row['bidVolume3']),
                            (row['bidPrice4'], row['bidVolume4']),
                            (row['bidPrice5'], row['bidVolume5'])]:
                take = int(min(this_qty - filled, max(0, vol)))
                if take <= 0: continue
                filled += take; cash += take*px
                if filled >= this_qty: break

        if filled > 0:
            trades += 1
            executed_qty += filled
            executed_value += cash
            vwap_num += filled * row['mid']
            vwap_den += filled
            remain -= filled

    completion = executed_qty / config.initial_inventory
    mid_last = processed_df.iloc[min(len(processed_df)-1, config.time_horizon-1)]['mid']
    if executed_qty > 0:
        avg_px = executed_value / executed_qty
        vwap = vwap_num / max(1, vwap_den)
        if config.side == 'buy':
            cost_bps = 10000*(avg_px - vwap)/vwap
            pv = -executed_value + executed_qty * mid_last
        else:
            cost_bps = 10000*(vwap - avg_px)/vwap
            pv = +executed_value - executed_qty * mid_last
    else:
        cost_bps, pv = 0.0, 0.0

    return {
        'completion': completion,
        'avg_cost_bps': cost_bps,
        'shortfall_bps': cost_bps,  # vs VWAP here
        'pv': pv,
        'num_trades': trades
    }