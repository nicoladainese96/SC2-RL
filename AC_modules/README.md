## Actor-Critic modules

Last update: 29/10/2020

List of files:
- Layers.py implements some non-trivial layers, some of which were never used because they are from the relational architecture, which I never had the time to 
implement.
- Networks.py implements various neural networks, which are used as building blocks of the AC architectures of A2C and IMPALA agents. 
Some networks also implement action sampling during the forward method.
- ActorCriticArchitecture.py contains full AC architectures with different design choices for the parameter networks and for their sampling techniques. 
These classes do not implement the loss computation and the updates.
- BatchedA2C.py contains various versions of the synchronous Advantage Actor-Critic (approximately one for each different version contained in the 
ActorCriticArchitecture module), which is also discussed in depth in the chapter 4.5 of the thesis. These classes are equipped to interact with the 
StarCraft environment through agent.step() and to compute the updates on the trajectories collected.
- IMPALA.py implements the Actor-Learner class both for monobeast (IMPALA_AC) and monobeast_v2 (IMPALA_AC_v2). Here the major challenge was to encode the 
compound actions from a batch of environments so that they would fit the buffers that then will be read by the learner, which has to be able to reconstruct each 
action and use it to compute the probability of choosing that action accordingly to its own parameters. 
