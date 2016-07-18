from dessn.chain.chain import ChainConsumer
from dessn.chain.examples.demo_data import get_data


data, _, _, _, parameters = get_data()

c = ChainConsumer().add_chain(data, parameters=parameters)
c.configure_general(flip=False, max_ticks=10)
c.plot(parameters=parameters[:2], filename="demoVarious3_Flip.png", figsize=(6, 6))
