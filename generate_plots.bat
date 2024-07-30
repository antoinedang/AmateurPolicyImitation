@echo off
setlocal enabledelayedexpansion

REM Define the environments and algorithms arrays
set envs=MountainCar Pendulum BipedalWalker LunarLander HalfCheetah Humanoid Walker2D
set algos=ppo sac td3 a2c

REM Loop through each environment
for %%e in (%envs%) do (
    REM Loop through each algorithm
    for %%a in (%algos%) do (
        python plot_evaluations.py policies/%%e/good_%%a/evaluations.npz policies/%%e/pure_random_%%a/evaluations.npz policies/%%e/bad_%%a/evaluations.npz policies/%%e/he_%%a_initialization/evaluations.npz policies/%%e/xavier_%%a_initialization/evaluations.npz policies/%%e/orthogonal_%%a_initialization/evaluations.npz --names Good Random Bad He Xavier Orthogonal --save plot_%%e_%%a.png
    )
)

endlocal
