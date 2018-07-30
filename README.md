# FastPareto
Fast implementations for finding the pareto front in a large set of points

## Done:
* A very quick python-numpy implementation for finding pareto points out of a set of 2d points:
  ```python
  import numpy as np
  from fastpareto import pareto
  import timeit
  testdata = np.random.randn(1000000,2) #one million test points
  t = timeit.timeit('pareto(testdata)', globals=globals(),number=10)/10
  print(str(t)+' seconds for finding pareto points in a set of 1 million 2d points')
  #>> 0.2959234861191362 seconds for finding pareto points in a set of 1 million 2d points
  ```
* A less quick python-numpy implemention for more than 2 costs:
  ```python
  import numpy as np
  from fastpareto import pareto
  import timeit
  testdata = np.random.randn(1000000,3) #one million test points
  t = timeit.timeit('pareto(testdata)', globals=globals(),number=10)/10
  print(str(t)+' seconds for finding pareto points in a set of 1 million 3d points')
  #>> 6.359629309689626 seconds for finding pareto points in a set of 1 million 3d points
  ```
  In this case, [Peter's answer on StackOverflow](https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python) seems to be quicker and is thus recommended.
  
## Todo:
* Faster python implementation for more than 2 costs
* Matlab port
## Legal:
* License (including disclaimers): see the LICENSE file, which is a copy of the GNU GPL license.
