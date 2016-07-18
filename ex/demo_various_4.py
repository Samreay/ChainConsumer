from dessn.chain.chain import ChainConsumer
from dessn.chain.examples.demo_data import get_data
import numpy as np

data, _, _, _, parameters = get_data()
c = ChainConsumer().add_chain(data, parameters=parameters)
c.configure_bar(summary=False).configure_contour(cloud=True, sigmas=np.linspace(0, 2, 10))
c.plot(filename="demoVarious4_ForceSummary.png")
