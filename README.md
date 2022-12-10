# Multiple Traveling Salesman Problem (MTSP)

https://www.tandfonline.com/doi/full/10.1080/21642583.2019.1674220

The path costs between the tags and between the robot positions and the
tags are received from the Topics PathCostsTag and PathCostsRobot and are
converted into the correct input format for the algorithm. The optimization
took place for 2000 generations with a population size of 10, a mutation prob-
ability of 0.05 and by generating two offsprings using binary tournament
selection, problem specific crossover and two different mutation algorithms
which get selected uniquely distributed. The optimized genome then is con-
verted to two lists of integers representing ordered drive plans containing
tag IDs for both robots. The tags assigned to robot r1d1 and r2d2 are pub-
lished to the topics r1d1DrivePlan and r2d2DrivePlan as DrivePlan messages.