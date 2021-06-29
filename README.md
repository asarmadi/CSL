# CSL
## Training-based Attack
In main_attack.py, an attack strategy is proposed for obtaining images that each party does not have access to. A network called generator is trained to generate those fake images. This attack stratesy has two step. The first is initialization in which the raw data that party 0 has access to is used to train G. In the next step a combination of part 0 features and shared features from party 1 is used to train the G further.
* Initialization

![Initialization](/Figs/G_init.png)

* Further Training

![Further Training](/Figs/architecture.png)

![Further Training with Regularization](/Figs/arch_l1.png)

## Optimization-based Attack
In opt_attack.py, an optimization-based attack is proposed in which the attacker tries to find a fake image that is classified to the target-label. The overall architecture is depicted in the following figure:

![Optimization-based](/Figs/optim_arch.png)

## Membership Inference Attack
In shokri_mi.py, the proposed membership inference attak in ["Comprehensive Privacy Analysis of Deep Learning"](https://www.comp.nus.edu.sg/~reza/files/Shokri-SP2019.pdf) is implemented for our problem.
