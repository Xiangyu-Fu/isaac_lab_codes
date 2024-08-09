# PONG with quadruped robot
## Description

## TODO
- [ ] find out how to create the pong env in isaac sim
- [ ] create the pong env in isaac sim
- [ ] find out how to train the pong agent in isaac sim

# Command
## Headless execution
isaaclab.bat -p user/PONG/skrl_pong_train.py --num_envs 64

isaaclab.bat -p source/standalone/workflows/skrl/train.py --task Isaac-Pong-Flat-GO-1-v0 --headless

## Headless execution with off-screen render
isaaclab.bat -p user/PONG/skrl_pong_train.py --num_envs 64 --headless --enable_cameras --video

## View the logs
isaaclab.bat -p -m tensorboard.main --logdir runs/torch/Isaac-Unitree

## Playing the trained agent
isaaclab.bat -p source/standalone/workflows/sb3/play.py --task Isaac-Pong-Flat-GO-1-v0 --num_envs 1 --use_last_checkpoint