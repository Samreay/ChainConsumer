from dessn.chain.chain import ChainConsumer
from dessn.chain.examples.demo_data import get_data

data, _, _, _, _ = get_data()
c = ChainConsumer()
c.add_chain(data)
c.configure_general(bins=1.5, kde=False)
c.plot(filename="demoVarious8_Extents.png", extents=[(-5, 5), (0, 15), (-3, 3), (-6, 6)])
