import os
import torch
import numpy as np
import pandas as pd
import warnings
import argparse
from typing import Dict
from config import Config
from envs import MarketEnv
from agent import PPOAgent
from utils import simulate_twap_taker, simulate_is_baseline, argss
from visualize import create_visualization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_system(data_path: str = "data/AA_Comdty_cpu",
                 n_episodes: int = 500,
                 save_interval: int = 50,
                 side: str = 'buy') -> Dict:
    
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved in '{checkpoint_dir}/' directory.")

    config = Config(side=side)
    if argss() is not None:
        args = argss()
        config.lr_actor = args.lr
        config.gamma = args.gamma
        config.n_epochs_per_update = args.n_epochs_per_update
        config.batch_size = args.batch_size
        config.buffer_size = args.buffer
        config.clip_ratio = args.clip_range 
        config.value_loss_coef = args.vf_coef
        config.entropy_beta = args.ent_coef  

    print(f"Loading data from {data_path}...")
    train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
    valid_data = pd.read_csv(os.path.join(data_path, "valid.csv"))
    test_data  = pd.read_csv(os.path.join(data_path, "test.csv"))
    print(f"Train: {len(train_data):,} | Valid: {len(valid_data):,} | Test: {len(test_data):,}")

    train_env = MarketEnv(train_data, config, mode='train', reward_type=args.reward_type)
    valid_env = MarketEnv(valid_data, config, mode='valid', reward_type=args.reward_type)
    test_env  = MarketEnv(test_data, config, mode='test', reward_type=args.reward_type)

    agent = PPOAgent(config)
    agent.network.to(device) 
    agent.network.train()

    history = {
        'train_rewards': [], 'train_completion': [], 'train_costs': [],
        'valid_rewards': [], 'valid_completion': [], 'valid_costs': [],
        'best_valid_reward': -float('inf'), 'best_valid_completion': 0
    }

    patience_counter = 0
    best_valid_score = -float('inf')

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    for ep in range(n_episodes):
        state = train_env.reset()
        ep_reward = 0.0
        ep_costs = []

        for _ in range(config.time_horizon):
            action, logp = agent.select_action(state, training=True)
            next_state, reward, done, info = train_env.step(action)
            agent.store_transition(state, action, reward, next_state, done, logp)
            ep_reward += reward
            if info['trade_executed']:
                ep_costs.append(info['trade_cost_bps'])
            state = next_state
            if done: break

        history['train_rewards'].append(ep_reward)
        history['train_completion'].append(info['completion_rate'])
        history['train_costs'].append(np.mean(ep_costs) if ep_costs else 0.0)

        if len(agent.buffer) >= config.buffer_size:
            _ = agent.update()

        if (ep+1) % 25 == 0:
            agent.network.eval()
            with torch.no_grad():
                state = valid_env.reset()
                v_reward, v_costs = 0.0, []
                for _ in range(config.time_horizon):
                    action, _ = agent.select_action(state, training=False)
                    next_state, reward, done, vinfo = valid_env.step(action)
                    v_reward += reward
                    if vinfo['trade_executed']:
                        v_costs.append(vinfo['trade_cost_bps'])
                    state = next_state
                    if done: break

            history['valid_rewards'].append(v_reward)
            history['valid_completion'].append(vinfo['completion_rate'])
            history['valid_costs'].append(np.mean(v_costs) if v_costs else 0.0)

            valid_score = v_reward + 100*vinfo['completion_rate']
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                history['best_valid_reward'] = v_reward
                history['best_valid_completion'] = vinfo['completion_rate']
                patience_counter = 0
                agent.save(os.path.join(checkpoint_dir, 'best_model.pth'))
                print("---New best model saved!---")
            else:
                patience_counter += 1

            print(f"\nEpisode {ep+1}/{n_episodes}")
            print(f"  Train - Reward: {np.mean(history['train_rewards'][-25:]):.1f} | "
                  f"Completion: {np.mean(history['train_completion'][-25:])*100:.1f}% | "
                  f"Cost: {np.mean(history['train_costs'][-25:]):.1f} bps")
            print(f"  Valid - Reward: {v_reward:.1f} | Completion: {vinfo['completion_rate']*100:.1f}% | "
                  f"Cost: {np.mean(v_costs) if v_costs else 0:.1f} bps")
            print(f"  Learning - LR: {agent.scheduler.get_last_lr()[0]:.6f}")

            if patience_counter >= config.patience:
                print(f"\n⚠ Early stopping triggered (patience={config.patience})")
                break
            agent.network.train()

        if (ep+1) % save_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f'checkpoint_ep{ep+1}.pth'))

    # ---------- Final testing ----------
    print("\n" + "="*60)
    print("FINAL TESTING")
    print("="*60)

    agent.load(os.path.join(checkpoint_dir, 'best_model.pth'))
    agent.network.eval()
    with torch.no_grad():
        state = test_env.reset()
        test_reward, test_costs, test_trades = 0.0, [], []
        for _ in range(config.time_horizon):
            action, _ = agent.select_action(state, training=False)
            next_state, reward, done, info = test_env.step(action)
            test_reward += reward
            if info['trade_executed']:
                test_costs.append(info['trade_cost_bps'])
                test_trades.append({'step': test_env.current_step,
                                    'size': info['order_size'],
                                    'filled': info['filled_qty'],
                                    'cost': info['trade_cost_bps']})
            state = next_state
            if done: break

    test_completion = info['completion_rate']
    avg_cost = np.mean(test_costs) if test_costs else 0.0

    # Shortfall vs VWAP
    if test_env.executed_qty > 0:
        avg_px = test_env.executed_value / test_env.executed_qty
        vwap = test_env.vwap_numerator / max(1, test_env.vwap_denominator)
        if config.side == 'buy':
            shortfall_bps = 10000*(avg_px - vwap)/vwap
        else:
            shortfall_bps = 10000*(vwap - avg_px)/vwap
    else:
        shortfall_bps = 0.0

    # Portfolio Value vs final mid (for traded quantity)
    final_mid = test_env.processed_data.iloc[min(test_env.current_idx, len(test_env.processed_data)-1)]['mid']
    if config.side == 'buy':
        portfolio_value = -test_env.executed_value + test_env.executed_qty * final_mid
    else:
        portfolio_value = +test_env.executed_value - test_env.executed_qty * final_mid

    # ---------- Baseline: TWAP (taker at L1) &IS----------
    twap_baseline = simulate_twap_taker(test_env.processed_data, config)
    is_baseline = simulate_is_baseline(test_env.processed_data, config)
    print(f"\nFINAL TEST RESULTS (side={config.side.upper()}):")
    print(f"  RL  - Reward: {test_reward:.1f} | Completion: {test_completion*100:.2f}% | "
          f"Avg Cost: {avg_cost:.2f} bps | Shortfall(VWAP): {shortfall_bps:.2f} bps | "
          f"PV: {portfolio_value:.2f} | Trades: {len(test_env.trades)}")
    print(f"  TWAP- Completion: {twap_baseline['completion']*100:.2f}% | "
          f"Avg Cost: {twap_baseline['avg_cost_bps']:.2f} bps | Shortfall(VWAP): {twap_baseline['shortfall_bps']:.2f} bps | "
          f"PV: {twap_baseline['pv']:.2f} | Trades: {twap_baseline['num_trades']}")
    print(f"  IS  - Completion: {is_baseline['completion']*100:.2f}% | "
          f"IS Cost: {is_baseline['avg_cost_bps']:.2f} bps | "
          f"PV: {is_baseline['pv']:.2f} | Trades: {is_baseline['num_trades']}")

    create_visualization(history, test_env, config, twap_baseline, is_baseline)

    return {
        'history': history,
        'test_reward': test_reward,
        'test_completion': test_completion,
        'test_cost': avg_cost,
        'test_shortfall': shortfall_bps,
        'test_trades': test_trades,
        'test_pv': portfolio_value,
        'is_baseline': is_baseline,
        'twap_baseline': twap_baseline

    }


if __name__ == "__main__":
    args = argss()
    if args.data == "cpu":
        data_path = "data/2019"
    elif args.data == "gpu":
        data_path = "data/2019"
    else:
        raise ValueError("Invalid data argument. Use 'cpu' or 'gpu'.")
    
    results = train_system(
        data_path=data_path,
        n_episodes=1500,
        save_interval=50,
        side='buy'
    )
