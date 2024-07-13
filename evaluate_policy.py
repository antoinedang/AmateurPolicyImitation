# take policy weights, environment
# output quality (average reward) and class balance


def evaluate_amateur_policy(model, env):
    reward_mean, reward_std = evaluate_policy(model, env)

    return (
        reward_mean,
        reward_std,
    )
