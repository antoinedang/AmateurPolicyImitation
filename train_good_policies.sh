algos=("ppo" "sac" "td3" "a2c")
envs_w_steps=("CartPole-v1,300_000,CartPole" "CliffWalking-v0,100_000,CliffWalking" "HalfCheetah-v4,1_000_000,HalfCheetah" "Humanoid-v4,2_000_000,Humanoid" "LunarLander-v2,300_000,LunarLander" "MountainCarContinuous-v0,300_000,MountainCar" "Walker2D-v4,1_000_000,Walker2D")

# For loop
for algo in "${algos[@]}"
do
    for pair in "${envs_w_steps[@]}"
    do
        IFS=',' read -r envid steps policy_folder <<< "$pair"
        echo "python3 train.py --algo $algo --env $envid --n-steps $steps --init policies/$policy_folder/good_$algo.pt"
        python3 train.py --algo $algo --env $envid --n-steps $steps --init policies/$policy_folder/good_$algo.pt
    done
done