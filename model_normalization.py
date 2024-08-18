def normalize_weights(state_dict, target_magnitude=0.3):
    normalized_state_dict = {}

    for key, weights in state_dict.items():
        if weights.ndimension() > 0:  # Skip if it's not a tensor (e.g., scalars)
            weights = weights.clone().detach()
            # Calculate the sum of the absolute values of the weights
            original_min = weights.min().item()
            weights = weights - original_min  # make minimum 0
            original_max = weights.max().item()
            weights = 2 * (weights / original_max)  # make maximum 2
            weights = weights - 1  # make maximum 1, minimum -1
            weights = (
                weights * target_magnitude
            )  # make range -target_magnitude to target_magnitude

            normalized_state_dict[key] = weights
        else:
            # If the entry is not a tensor (e.g., a scalar), just copy it as is
            normalized_state_dict[key] = weights

    return normalized_state_dict
