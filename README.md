The Petrol Station Replenishment Problem (PSRP) involves distributing fuel from depots to stations using multi-compartment vehicles over a multi-day horizon. Traditional methods are computationally intensive and focus on single-day solutions, limiting efficiency. This paper proposes a reinforcement learning-based approach to optimize routing and inventory management, reducing costs and improving operational efficiency. By leveraging learned patterns, the model adapts to dynamic constraints and eliminates the need for frequent recalculations, offering a scalable solution to PSRP.

# Requirements

* gym
* torch
* gymnasium
* sb3_contrib
* tensorboard
* torch_geometric
* stable-baselines3[extra]

# Example usage

## Model Training

```bash
python train.py --n 3 --veh 2 --n_steps 500 --timesteps 10000
```

### Result

Run `tensorboard --logdir ppo_tensorboard` to monitor the training progress and view the prediction results.

![Описание изображения](Images/example_tensorboard.png)

## Model Eval

```bash
python eval.py --n 3 --veh 2 --n_steps 500
```

### Result

![Описание изображения](Images/rewards_10_500_2.png)


## Possible parameters:

Ниже представлены примеры запуска моделей, которые находятся в `data_pkl`
```
 --n [3, 4, 5, 6, 7, 8, 9, 10] - Number of nodes
 --veh [2, 4, 6] - Number of vehicles
 --n_steps [250, 500, 1000] - Number of steps
 --timesteps - Number of training steps
 --pre_train [True, False] - Loading a pre-trained model
```

---

# Data for Model

The data is provided in a `.pkl` format, containing a `tuple` of 11 attributes. Each attribute is represented as a `torch.Tensor`. These data are used to train the model and define the problem parameters. Below is the data format description:

1. **positions:**  
   A 2D tensor of size `(n, 2)` representing the coordinates of the points.  
   Example:
   ```python
   tensor([[274., 149.],
           [ 82., 223.],
           [ 94.,  49.]], dtype=torch.float64)
   ```

2. **weight_matrixes:**  
   A square matrix of size `(n, n)` containing weights (e.g., distances between points).  
   Example:
   ```python
   tensor([[    0., 12300., 12300.],
           [12300.,     0., 10440.],
           [12300., 10440.,     0.]], dtype=torch.float64)
   ```

3. **daily_demands:**  
   A 3D tensor of size `(d, n, 2)`, where `d` is the number of days, `n` is the number of points, and `2` represents demand parameters.  
   Example:
   ```python
   tensor([[[ 0.,  0.], 
            [10., 10.],
            [10., 10.]],
           ...
           [[ 0.,  0.],
            [10., 10.],
            [10., 10.]]], dtype=torch.float64)
   ```

4. **depots:**  
   A 1D tensor specifying the node index of the depot.  
   Example:
   ```python
   tensor([0.], dtype=torch.float64)
   ```

5. **working_time:**  
   A 1D tensor representing the working time in seconds.  
   Example:
   ```python
   tensor([32400., 32400.], dtype=torch.float64)
   ```

6. **restriction_matrix:**  
   A 2D integer tensor describing restrictions or constraints.  
   Example:
   ```python
   tensor([[0, 0, 0],
           [0, 0, 0]], dtype=torch.int32)
   ```

7. **service_times:**  
   A 1D tensor with the service time required at each point.  
   Example:
   ```python
   tensor([900., 900., 900.], dtype=torch.float64)
   ```

8. **min_capacities:**  
   A 2D tensor of size `(n, 2)` representing the minimum capacity limits that cannot be exceeded downward.  
   Example:
   ```python
   tensor([[0., 0.],
           [5., 5.],
           [5., 5.]], dtype=torch.float64)
   ```

9. **max_capacities:**  
   A 2D tensor of size `(n, 2)` representing the maximum capacity limits.  
   Example:
   ```python
   tensor([[ 0.,  0.],
           [95., 95.],
           [95., 95.]], dtype=torch.float64)
   ```

10. **init_capacities:**  
    A 2D tensor of size `(n, 2)` representing the initial capacities.  
    Example:
    ```python
    tensor([[ 0.,  0.],
            [50., 50.],
            [50., 50.]], dtype=torch.float64)
    ```

11. **vehicle_compartments:**  
    A 3D tensor of size `(v, n, 2)`, where `v` is the number of vehicles.  
    Example:
    ```python
    tensor([[[50., 50.],
             [50., 50.],
             [50., 50.]],
            [[50., 50.],
             [50., 50.],
             [50., 50.]]], dtype=torch.float64)
    ```