# CSL
In main_attack.py, an attack strategy is proposed for obtaining images that each party does not have access to. A network called generator is trained to generate those fake images. This attack stratesy has two step. The first is initialization in which the raw data that party 0 has access to is used to train G. In the next step a combination of part 0 features and shared features from party 1 is used to train the G further.
* Initialization

!(/Figs/G_init.png)

* Further Training

!(/Figs/architecture.png)

!(/Figs/arch_l1.png)
