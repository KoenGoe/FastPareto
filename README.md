# FastPareto
Fast implementations for finding pareto front in set of points

## Done:
* A very quick python-numpy implementation for finding pareto points out of a set of 2d points:
  ```python
  import numpy as np
  from fastpareto import pareto
  import timeit
  testdata = np.random.randn(1000000,2) #one million test points
  t = timeit.timeit('pareto(testdata)', globals=globals(),number=10)/10
  print(str(t)+' seconds for finding pareto points in a set of 1 million 2d points')
  #>> 0.2959234861191362 seconds for finding pareto points in a set of 1 million 2d points```
* A far less but still rather quick python-numpy implemention for more than 2 costs:
  ```python
  import numpy as np
  from fastpareto import pareto
  import timeit
  testdata = np.random.randn(1000000,3) #one million test points
  t = timeit.timeit('pareto(testdata)', globals=globals(),number=10)/10
  print(str(t)+' seconds for finding pareto points in a set of 1 million 3d points')
  #>> 6.359629309689626 seconds for finding pareto points in a set of 1 million 3d points```
  
## Todo:
* Faster python implementation for more than 2 costs
* Matlab port
