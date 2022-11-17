import statistics


# evaluate the agent on the env for number_episodes
def evaluate(agent, env, number_episodes, verbose=False):
    # init empty record
    rewards = []
    agent.set_exploit(True)

    # for N episodes
    for episode in range(number_episodes):
        if verbose:
            print("\tTesting episode ", episode)

        # reset and begin record keeping
        obs, info = env.reset()
        done, truncated = False, False
        current_reward = 0

        # Iterate through episode
        while not done and not truncated:
            action = agent(obs)
            nobs, reward, done, truncated, info = env.step(action)
            current_reward += reward
            obs = nobs

        # after episode, add to record
        rewards.append(current_reward)

    if verbose:
        print("\tCompleted Test")
    agent.set_exploit(False)
    # return mean and std deviation reward
    return statistics.mean(rewards), statistics.stdev(rewards)
