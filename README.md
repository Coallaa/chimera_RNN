# chimera_RNN

### Setup and Environment

```
conda env create -f env.yml
```



### Program Input

- Target file: (--input)
  - ../input/target.pkl

- Initial Parameters: (--origin)

  - ../input/origin.txt

- Dictionary to store the output files: (--dic)

  - ../test_case/ 
  - don't forget last '/'

- Device parameter: (--device)

  - gpu/cpu

- Training parameters: (--rounds [a,b,c])

  - a: num of pre-training rounds
  - b: num of training rounds
  - c: num of post-training rounds

- Model parameters:

  - num_neuron 1500 (--num_neuron)

  - group_info: an array of grouping information [3,3] 

    (--group_info [x_0,x_1,....]  **x** how many nodes in each group)

  - dt: learning rate -> originally set to time step 0.1 (--dt)

  - sparse_param: pre-train weight sparse rate 0.1 (--sparse_param)

  - lam: controls the rate of error ( $P_0 = I_n/\lambda$) 1 (--lam)

  - G: the scale parameter of omega, control the chaotic behaviour 1.5 (--G)

  - Q: the scale parameter of eta. 1 (--Q)

  - $z_{t+1} = z_t + dt\left( -z_t + \left(G\omega^0+Q\eta d_t^{\top}\right) \tanh{z_t}\right)$

- Evaluation parameters:
  - compare_round: compare result after x rounds post train. (--compare_round)
  - statistic_window: length of statistic calculation window. (--statistic_window)
  - tar_out_window: length of tar_out plot window. ('--tar_out_window)
  - l2_error_window: length of l2 error plot window. (--l2_error_window)



### Input format

e.g. 

```
# use default parameter
python automatic/main.py
# add whatever parameters
python automatic/main.py --device cpu --num_neuron 1000
```



### Program output

- Output files: 

  - predict.pkl: The output of the network whose shape is N*M.

    N is the sum of training parameters and M is number of nodes.

  - error.pkl: The error of the network whose shape is also N*M.

- Statistic results: result.txt

  The l2 error with size 12 (n\*2\*2)

  - n: num_of_group + 1 e.g. $\theta,\phi,\theta+\phi$

  - 2: just finish training/after 300k post-train

  - 2: mean, var

  e.g. when n is 2, its format is [x_0, ..., x_11]

  - x_0: (all_group, just finish, mean)
  - x_1: (all_group, just finish, var)
  - x_2: (all_group, after 300k, mean)
  - x_3: (all_group, after 300k, var)
  - x_4: (group0, just finish, mean)
  - x_5: (group0, just finish, var)
  - ...
  - x_10:  (group2, after 300k, mean)
  - x_11:  (group2, after 300k, var)

- Image 4: 2*2

  - 2: tar_out / l2 error

  - 2: just finish training/after 300k post-train

