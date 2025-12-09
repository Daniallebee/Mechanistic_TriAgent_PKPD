"""
Train a simple PPO agent and validate performance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from triagent_controller.rl_environment import AnesthesiaEnv

class MetricsCallback(BaseCallback):
    """Custom callback for tracking training metrics"""
    
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.bis_performance = []
        self.map_performance = []
    
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            
            # Get episode data from monitor
            episode_reward = self.locals.get("episode_rewards")[0]
            episode_length = self.locals.get("episode_lengths")[0]
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                print(f"Episode {len(self.episode_rewards)}: "
                      f"Reward={episode_reward:.2f}, "
                      f"Length={episode_length}")
        
        return True

def train_and_validate():
    """Train PPO agent and validate performance"""
    
    print("Creating environment...")
    # Create and wrap environment
    env = Monitor(AnesthesiaEnv(episode_length_minutes=30))
    eval_env = AnesthesiaEnv(episode_length_minutes=30)
    
    print("Initializing PPO agent...")
    # Create PPO model with tuned hyperparameters for this task
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )
    
    # Callback for metrics
    callback = MetricsCallback(verbose=1)
    
    print("\nTraining for 50,000 timesteps...")
    print("This should take about 5-10 minutes...")
    
    # Train
    model.learn(total_timesteps=50000, callback=callback)
    
    # Save model
    model.save("ppo_anesthesia_simple")
    print("\n✓ Model saved as 'ppo_anesthesia_simple'")
    
    # Plot training progress
    if len(callback.episode_rewards) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Smooth rewards
        window = min(20, len(callback.episode_rewards) // 5)
        if window > 0:
            smoothed = np.convolve(callback.episode_rewards, 
                                   np.ones(window)/window, mode='valid')
            ax1.plot(smoothed, label='Smoothed')
        ax1.plot(callback.episode_rewards, alpha=0.3, label='Raw')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(callback.episode_lengths)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Duration')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        print("✓ Training progress saved as 'training_progress.png'")
    
    # Evaluate trained agent
    print("\n" + "=" * 50)
    print("EVALUATING TRAINED AGENT")
    print("=" * 50)
    
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, return_episode_rewards=False
    )
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Detailed evaluation
    evaluate_agent_performance(model, eval_env)
    
    return model

def evaluate_agent_performance(model, env):
    """Detailed performance evaluation"""
    
    print("\nDetailed Performance Analysis:")
    print("-" * 30)
    
    # Run one episode with trained agent
    obs = env.reset()
    episode_data = []
    
    for _ in range(env.episode_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        episode_data.append({
            'bis': info['bis'],
            'map': info['map'],
            'hr': info['hr'],
            'u_propo': info['drug_rates'][0],
            'u_remi': info['drug_rates'][1],
            'u_nore': info['drug_rates'][2],
            'reward': reward
        })
        
        if done:
            break
    
    # Calculate metrics
    bis_values = [d['bis'] for d in episode_data]
    map_values = [d['map'] for d in episode_data]
    rewards = [d['reward'] for d in episode_data]
    
    bis_in_target = sum(40 <= b <= 60 for b in bis_values) / len(bis_values) * 100
    map_safe = sum(m >= 65 for m in map_values) / len(map_values) * 100
    
    print(f"BIS in target (40-60): {bis_in_target:.1f}%")
    print(f"MAP safe (≥65): {map_safe:.1f}%")
    print(f"Average BIS: {np.mean(bis_values):.1f}")
    print(f"Average MAP: {np.mean(map_values):.1f}")
    print(f"Total reward: {sum(rewards):.1f}")
    
    # Validation criteria
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    
    validations = [
        ("BIS control", bis_in_target > 60, f"{bis_in_target:.1f}% > 60%"),
        ("MAP safety", map_safe > 80, f"{map_safe:.1f}% > 80%"),
        ("Average BIS", 45 <= np.mean(bis_values) <= 55, 
         f"{np.mean(bis_values):.1f} in [45, 55]"),
        ("Average MAP", np.mean(map_values) >= 70, 
         f"{np.mean(map_values):.1f} >= 70"),
    ]
    
    for name, passed, condition in validations:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name} ({condition})")
    
    # Plot final episode
    plot_episode_comparison(episode_data)
    
    return bis_in_target, map_safe

def plot_episode_comparison(episode_data):
    """Plot trained agent performance"""
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    fig.suptitle('Trained PPO Agent Performance', fontsize=14)
    
    time_min = np.arange(len(episode_data)) * 5 / 60
    
    # BIS
    bis = [d['bis'] for d in episode_data]
    axes[0, 0].plot(time_min, bis, 'b-')
    axes[0, 0].fill_between(time_min, 40, 60, alpha=0.2, color='green', 
                            label='Target Range')
    axes[0, 0].set_ylabel('BIS')
    axes[0, 0].set_title('Bispectral Index')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MAP
    map_vals = [d['map'] for d in episode_data]
    axes[1, 0].plot(time_min, map_vals, 'r-')
    axes[1, 0].axhline(y=65, color='green', linestyle='--', 
                      label='Safety Threshold')
    axes[1, 0].set_ylabel('MAP (mmHg)')
    axes[1, 0].set_title('Mean Arterial Pressure')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Heart Rate
    hr = [d['hr'] for d in episode_data]
    axes[2, 0].plot(time_min, hr, 'g-')
    axes[2, 0].set_ylabel('HR (bpm)')
    axes[2, 0].set_xlabel('Time (min)')
    axes[2, 0].set_title('Heart Rate')
    axes[2, 0].grid(True)
    
    # Drug rates
    u_propo = [d['u_propo'] for d in episode_data]
    axes[0, 1].plot(time_min, u_propo)
    axes[0, 1].set_ylabel('Propofol (mg/s)')
    axes[0, 1].set_title('Propofol Infusion')
    axes[0, 1].grid(True)
    
    u_remi = [d['u_remi'] for d in episode_data]
    axes[1, 1].plot(time_min, u_remi)
    axes[1, 1].set_ylabel('Remifentanil (µg/s)')
    axes[1, 1].set_title('Remifentanil Infusion')
    axes[1, 1].grid(True)
    
    u_nore = [d['u_nore'] for d in episode_data]
    axes[2, 1].plot(time_min, u_nore)
    axes[2, 1].set_ylabel('Norepinephrine (µg/s)')
    axes[2, 1].set_xlabel('Time (min)')
    axes[2, 1].set_title('Norepinephrine Infusion')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('trained_agent_performance.png')
    print("\n✓ Performance plot saved as 'trained_agent_performance.png'")

if __name__ == "__main__":
    # Check dependencies
    try:
        import stable_baselines3
        print("✓ Dependencies OK")
    except ImportError:
        print("Please install: pip install stable-baselines3")
        exit(1)
    
    # Train and validate
    model = train_and_validate()
    print("\n✅ Training and validation complete!")