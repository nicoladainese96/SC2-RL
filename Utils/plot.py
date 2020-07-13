import numpy as np
import matplotlib.pyplot as plt

def plot_results(results, moving_average=False, average_window=100):
    
    score, asymptotic_score, asymptotic_err, trained_agent, time_profile, losses = results
    t_play = time_profile[:,0].mean()
    t_update = time_profile[:,1].mean()

    print("Average time for playing one episode: %.2f s"%t_play)
    print("Average time for updating the agent: %.2f s"%t_update)
    print("Asymptotic score: %.3f +/- %.3f"%(asymptotic_score, asymptotic_err))
    
    if moving_average:
        n_epochs = np.arange(100, len(score))
    else:
        n_epochs = np.arange(len(score))
        
    ### plot score ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(score[i:i+100]) for i in range(len(score)-100)])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, score)
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.show()
    
    ### plot critic loss ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['critic_losses'][i:i+100]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, losses['critic_losses'])

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Critic loss", fontsize=16)
    plt.show()

    ### plot actor loss ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['actor_losses'][i:i+100]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, losses['actor_losses'])

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Actor loss", fontsize=16)
    plt.show()
    
    ### plot entropy ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['entropies'][i:i+100]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, -average_score)
    else:
        plt.plot(n_epochs, -np.array(losses['entropies']))

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Entropy term", fontsize=16)
    plt.show()
    
    
def plot_session(score, losses, moving_average=False, average_window=100):
    
    if moving_average:
        n_epochs = np.arange(100, len(score))
    else:
        n_epochs = np.arange(len(score))
        
    ### plot score ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(score[i:i+100]) for i in range(len(score)-100)])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, score)
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.show()
    
    ### plot critic loss ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['critic_losses'][i:i+100]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, losses['critic_losses'])

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Critic loss", fontsize=16)
    plt.show()

    ### plot actor loss ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['actor_losses'][i:i+100]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, losses['actor_losses'])

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Actor loss", fontsize=16)
    plt.show()
    
    ### plot entropy ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['entropies'][i:i+100]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, -average_score)
    else:
        plt.plot(n_epochs, -np.array(losses['entropies']))

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Entropy term", fontsize=16)
    plt.show()
    
def plot_bA2C_session(score, losses, unroll_length, test_interval, moving_average=False, average_window=100):
    
    if moving_average:
        n_epochs = np.arange(100, len(score))*unroll_length*test_interval
    else:
        n_epochs = np.arange(len(score))*unroll_length*test_interval
        
    ### plot score ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(score[i:i+100]) for i in range(len(score)-100)])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, score)
    plt.xlabel("Number of steps", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.show()
    
    if moving_average:
        n_epochs = np.arange(100, len(losses['critic_losses']))*unroll_length
    else:
        n_epochs = np.arange(len(losses['critic_losses']))*unroll_length
        
    ### plot critic loss ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['critic_losses'][i:i+100]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, losses['critic_losses'])

    plt.xlabel("Number of steps", fontsize=16)
    plt.ylabel("Critic loss", fontsize=16)
    plt.show()

    ### plot actor loss ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['actor_losses'][i:i+100]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, losses['actor_losses'])

    plt.xlabel("Number of steps", fontsize=16)
    plt.ylabel("Actor loss", fontsize=16)
    plt.show()
    
    ### plot entropy ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['entropies'][i:i+100]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, -average_score)
    else:
        plt.plot(n_epochs, -np.array(losses['entropies']))

    plt.xlabel("Number of steps", fontsize=16)
    plt.ylabel("Entropy term", fontsize=16)
    plt.show()