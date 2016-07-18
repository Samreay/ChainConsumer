from dessn.chain.chain import ChainConsumer
from dessn.chain.examples.demo_data import get_data


data, _, _, _, parameters = get_data()
c = ChainConsumer().add_chain(data, parameters=parameters)
c.plot(parameters=parameters[:3], filename="demoVarious2_SelectParameters.png")
