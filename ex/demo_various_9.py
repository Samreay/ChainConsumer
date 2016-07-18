from dessn.chain.chain import ChainConsumer
from dessn.chain.examples.demo_data import get_data

data, _, _, _, parameters = get_data()
c = ChainConsumer()
c.add_chain(data)
c.configure_general(kde=True)
c.plot(filename="demoVarious9_kde.png")
