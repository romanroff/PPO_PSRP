## Model Training

```bash
python train.py --n 3 --veh 2 --n_steps 500 --timesteps 10000
```

## Model Eval

```bash
python eval.py --n 3 --veh 2 --n_steps 500
```
Possible parameters:
```
 --n [3, 4, 5, 6, 7, 8, 9, 10] - Number of nodes
 --veh [2, 4, 6] - Number of vehicles
 --n_steps [250, 500, 1000] - Number of steps
 --timesteps - Number of training steps
 --pre_train [True, False] - Loading a pre-trained model
```

Run `tensorboard --logdir ppo_tensorboard` to monitor the training progress and view the prediction results.