import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def smooth(data, weigth=0.9):
    """用于平滑曲线，类似于Tensorboard中的smooth
    
    Args:
        data(List): 用于平滑的数组
        weigth(Float): 平滑权重，0-1之间，值越大越平滑，一般取0.9
    
    Returns:
        smoothed(List): 平滑后的数组
    """
    last = data[0] # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weigth + (1 - weigth) * point # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val # Array is smoothed so the last value is smoothed_val
    return smoothed

def plot_rewards(rewards, cfg, tag="train"):
    sns.set_theme()
    plt.figure()
    plt.title(f"{tag}ing curve on {cfg.device} if {cfg.algo_name} for {cfg.env_name}")
    plt.plot(rewards, label="rewards")
    plt.plot(smooth(rewards), label="smoothed rewards")
    plt.legend()
    plt.show()