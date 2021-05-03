# CellFreeCLCA_RL
The simulation of paper: Joint Cooperation Clustering and Content Caching in Cell-Free Massive MIMO Networks.

A. Directory Path
  1. Topology:                data/{topology name}/Topology 
  2. Model:                   data/{topology name}/Model
  3. [Figure] performance:    data/{topology name}/EvaluationPhase
  4. [Figure] Sampled policy: data/{topology name}/EVSampledPolicy

B. Code mainbody: 
  1. newSimulate.py
  2. newENV.py
  3. newDDPG.py
  
C. Dependency:
  1. newSimulate.py <- newENV.py
  2. newSimulate.py <- newDDPG.py

D. Standard model training and evaluation procedure:(run newSimulate.py)
  1. Train model using trainModel()
  2. Evaluate trained model using evaluateModel()
  3. plot evaluation result using plotHistory(), the function can plot EE/HR/TP/Psys/MCAP/MCCPU
  4. plot policy sampled in evaluation phase using plot_UE_BS_distribution_Cache()

E. Generate arbitrary topology:
  1. in newENV(), declare BS by set parameter loadENV=False
    EX: env = BS(nBS=10,nUE=5,nMaxLink=3,nFile=20,nMaxCache=2,loadENV = False,SEED=0,obsIdx=1)
    It will produce a new topology in data/10.5.20.2/Topology in pkl form and png preview
    
F. 
    
