# An adaptive hierarchical energy management strategy
This project contains the source code of an adaptive hierarchical EMS combining heuristic equivalent consumption minimization strategy (ECMS) knowledge and deep deterministic policy gradient (DDPG). It can be used to reproduce the results described in the paper "An adaptive hierarchical energy management strategy for hybrid electric vehicles combining heuristic engineering domain knowledge and data-driven deep reinforcement learning, submitted to IEEE Transactions on Transportation Electrification".  
<div align=center>
<img src="https://user-images.githubusercontent.com/91611499/136344352-a87be4d7-2279-4bc7-ba96-8fa541d65633.png" width = "400"  alt="schematic diagram" align=center />
</div>
<div align=center>
Figure.1 An adaptive hierarchical energy management strategy combining heuristic ECMS and data-driven DDPG
</div>  

## Installation Dependencies:
- Python3.6  
- Tensorflow1.12  
- Matlab2019B   

## How to run:
1. Add the folder which extracted from Proposed strategy.rar to the environment path of MATLAB.
2. Put 'main.py' in 'main/system' then run it.
3. Observe the printed results of each episode.

## Main files:
- main.py: The main program containing the source of the proposed algorithm.  
- Proposed strategy\main\System\HevP2ReferenceApplication: The simulink simulator of the hybrid electric vehicle.  
- Proposed strategy\main\System\Interaction.m: The interactive Matlab Engine API for the main Python program.  
- Proposed strategy\main\System\Initialize_simulink.m: Use this sentence to initialize Matlab Engine API for the main Python program and restart the simulation model after the end of the previous episode. (Some MATLAB functions return no output arguments. If the function returns no arguments, set nargout to 0)  
<div align=center>
<img src="https://user-images.githubusercontent.com/91611499/136706160-91f3f3c4-6982-44cb-8803-f3ae250abf16.png" width = "600"  alt="flow chart" align=center />
</div>
<div align=center>
Figure.2 Flow chart
</div>  

## Calling Matlab/Simulink from Python
To start the Matlab engine within a Python session, you first must install the engine API as a Python package. MATLAB provides a standard Python setup.py file for building and installing the engine using the distutils module. You can use the same setup.py commands to build and install the engine on Windows, Mac, or Linux systems.  
Each Matlab release has a Python setup.py package. When you use the package, it runs the specified Matlab version. To switch between Matlab versions, you need to switch between the Python packages. For more information, see https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html    
Use follows sentence to import matlab.engine module and start the Matlab engine:  
```python
import matlab.engine
engine = matlab.engine.start_matlab()  
```    
Use this sentence to initialize Matlab Engine API for the main Python program and restart the simulation model after the end of the previous episode. (Some MATLAB functions return no output arguments. If the function returns no arguments, set nargout to 0)  
```python
engine.Initialize_simulink(nargout=0)
```  
Use this sentence to interact between Python and Matlab/Simulink. (You can call any Matlab function directly and return the results to Python. When you call a function with the engine, by default the engine returns a single output argument. If you know that the function can return multiple arguments, use the nargout argument to specify the number of output arguments.)  
```python
SOC, ReqPow, Clock, EquFuelCon= engine.Interaction(action, nargout=4)
```  
This sentence realize the interaction between Python and Matlab/simulink. Use this sentence to transfer action from DDPG agent to simulation model of Simulink. Then transfer simulation data from simulation model back to DDPG agent of Python.    

- SOC: Battery SOC.
- ReqPow: Required power. 
- Clock: Simulation time. 
- EquFuelCon: Equivalant fuel consumption.  
- action: action of DDPG agent.  

Note that in the proposed algorithm, the SOC, the required power and the last control action is chosen as state variables, the EF is the control action and the immediate reward is defined by the function of the deviation of the current SOC from the target SOC.  

## Hyperparameter：
<div align=center>  
  
| Parameter  |  Value |
| :------------: | :------------: |
| Number of hidden layers  | 3  |
| Neurons in each hidden layers  |   120|
| Activation function  | relu  |
|  Learning rate for actor | 0.0001  |
| Learning rate for critic  | 0.0002  |
|  Reward discount factor |  0.9 |
|  Soft replacement factor |  0.001 |
| Replay memory size  |  10000 |
| Mini-batch size  |  64 |  
  
</div>

## Attention:
The environment runs in FTP75 condition by default. If you want to change it, you need to open 'main\System\HevP2ReferenceApplication' and install drive cycle source toolbox, then change the running time in Simulink and main.py file.

## Performence
We train the reinforcement learning agent to minimize the fuel consumption using the proposed strategy. Figure.3 shows the SOC sustenance behavior between the proposed startegy and the other three benchmark algorithms.

<div align=center>
<img src="https://user-images.githubusercontent.com/91611499/136705440-e641d14f-2268-46da-83f1-153b69d7662c.png" width = "600"  alt="flow chart" align=center />
</div>
<div align=center>
Figure.3 SOC trajectories between the optimized proposed strategy and benchmark strategies
</div>    

Figure.4 shows the different engine working areas in different control strategies. Although the SOC trajectories differ considerably between the proposed and the DP-based strategy, the engine working areas under the two strategies locate in similar higher fuel efficiency regions more frequently, compared to the other benchmark strategies.  

<div align=center>
<img src="https://user-images.githubusercontent.com/91611499/136705690-8e8cda10-efaf-4052-bd18-4afd0551081e.png" width = "600"  alt="flow chart" align=center />
</div>
<div align=center>
Figure.4 Engine working areas for different control strategies
</div>    
