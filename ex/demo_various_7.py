from dessn.chain.chain import ChainConsumer
from dessn.chain.examples.demo_data import get_data

data, data2, data3, data4, parameters = get_data()
c = ChainConsumer()
c.add_chain(data, name="A")
c.add_chain(data2, name="B")
c.add_chain(data3, name="C")
c.add_chain(data4, name="D")
c.configure_general(bins=150, serif=False, rainbow=True)
c.plot(filename="demoVarious7_Rainbow.png")
