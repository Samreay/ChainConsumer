from dessn.chain.chain import ChainConsumer
from dessn.chain.examples.demo_data import get_data

data, data2, _, _, parameters = get_data()
c = ChainConsumer().add_chain(data, parameters=parameters).add_chain(data2)
c.configure_general(colours=["#B32222", "#D1D10D"])
c.configure_contour(contourf=True, contourf_alpha=0.5)
c.plot(filename="demoVarious5_CustomColours.png")


