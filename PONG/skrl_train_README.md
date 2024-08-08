# Command
## Headless execution
isaaclab.bat -p user/PONG/skrl_pong_train.py --num_envs 64

isaaclab.bat -p source/standalone/workflows/skrl/train.py --task Isaac-Pong-Flat-GO-1-v0 --headless

## Headless execution with off-screen render
isaaclab.bat -p user/PONG/skrl_pong_train.py --num_envs 64 --headless --enable_cameras --video

## View the logs
isaaclab.bat -p -m tensorboard.main --logdir runs/torch/Isaac-Velocity-Anymal-C-v0

## Playing the trained agent
isaaclab.bat -p source/standalone/workflows/sb3/play.py --task Isaac-Cartpole-v0 --num_envs 32 --use_last_checkpoint