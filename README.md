# [ChainConsumer](https://samreay.github.io/ChainConsumer)

[![Build Status](https://img.shields.io/travis/Samreay/ChainConsumer.svg?style=flat-square)](https://travis-ci.org/Samreay/ChainConsumer)


A new library to consume your fitting chains! Produce likelihood surfaces,
plot your walks to check convergence, or even output a LaTeX table of the
marginalised parameter distributions with uncertainties and significant
figures all done for you!

[Click through to the online documentation](https://samreay.github.io/ChainConsumer)

```python
import numpy as np
from chain_consumer import ChainConsumer


mean = [0.0, 4.0]
data = np.random.multivariate_normal(mean, [[1.0, 0.7], [0.7, 1.5]], size=100000)

c = ChainConsumer()
c.add_chain(data, parameters=["$x_1$", "$x_2$"])
c.plot(filename="example.png", figsize="column", truth=mean)
```


![Example plot](example.png)

