"""
=========
Something
=========

Something else

"""

from chain_consumer import ChainConsumer


data, _, _, _, parameters = get_data()
c = ChainConsumer().add_chain(data, parameters=parameters)
c.configure_general(plot_hists=False)
# c.plot(filename="demoVarious1_NoHist.png")
c.plot()


