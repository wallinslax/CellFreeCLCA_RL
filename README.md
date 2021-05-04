# CellFreeCLCA_RL

A. Directory Path
```
data 
├── 4.4.5.2               # Topology with 4 UE/4 AP/5 File/2 Cache 
│   ├── BF                # Brute Force result (pkl and png)
│   ├── Topology          # Topology (pkl and png)
│   ├── Preview           # In Training phase, we can preview the performance result in this folder, the result will update each 1000 time slot
│   ├── Model             # After training is done, the trained model will be saved in this folder. In single actor case, there are 2 model (actor and critic).
│   ├── EvaluationPhase   # After evaluation is done, plotHistory() will plot the performnace result and save the corresponding figures in this folder
│   └── EVSampledPolicy   # In evaluation phase, we will take a policy snapshot in half length of evaluation, the snapshot will save in this folder and be plotted via plot_UE_BS_distribution_Cache()
└── 10.5.20.2
    └── (same as above)
```


B. Code mainbody: 
  1. newSimulate.py
  2. newENV.py
  3. newDDPG.py

The code is run under Pytorch:1.8.0.dev20201027 and Python 3.8.5.
  
C. Dependency:
  1. newSimulate.py <- newENV.py
  2. newSimulate.py <- newDDPG.py

D. Standard model training and evaluation procedure:(run newSimulate.py)
  1. Train model using trainModel()
  2. Evaluate trained model using evaluateModel()
  3. plot evaluation result using plotHistory(), the function can plot EE/HR/TP/Psys/MCAP/MCCPU
  4. plot policy sampled in evaluation phase using plot_UE_BS_distribution_Cache()

E. Generate arbitrary topology:\
  In newENV(), declare class BS by set parameter loadENV=False. For example, 
  ```
  env = BS(nBS=10,nUE=5,nMaxLink=3,nFile=20,nMaxCache=2,loadENV = False,SEED=0,obsIdx=1)
  ```
  It will produce a new topology in data/10.5.20.2/Topology in pkl form and png preview
    
F. Derve Brute Force (BF) result (Direct example)\
In newENV(), run
```
env=BS(nBS=10,nUE=5,nMaxLink=nMaxLink,nFile=20,nMaxCache=2,loadENV = True,SEED=0)
env.getOptEE_BF(isSave=True)
```

    
