# CellFreeCLCA_RL
The simulation of paper: Joint Cooperation Clustering and Content Caching in Cell-Free Massive MIMO Networks.

A. Directory Path
  1.Topology:           data/{topology name}/Topology 
  2.Model:              data/{topology name}/Model
  3.performance figure: data/{topology name}/EvaluationPhase
  4.Sampled policy:     data/{topology name}/EVSampledPolicy
  5.

B. Code mainbody: 
  1. newSimulate.py
  2. newENV.py
  3. newDDPG.py
  
C. Dependency:
  newSimulate.py <- newENV.py
  newSimulate.py <- newDDPG.py

D. Standard model training and evaluation procedure:
  1. Train model using trainModel()
  2. Evaluate trained model using evaluateModel()
  3. plot evaluation result using plotHistory(), the function can plot EE/HR/TP/Psys/MCAP/MCCPU
  4. plot policy sampled in evaluation phase using plot_UE_BS_distribution_Cache()
