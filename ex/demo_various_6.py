from dessn.chain.chain import ChainConsumer
from dessn.chain.examples.demo_data import get_data

data, _, _, _, parameters = get_data()
c = ChainConsumer().add_chain(data, parameters=parameters)
c.plot(filename="demoVarious6_TruthValues.png", truth=[0.0, 5.0, 0.0, 0.0])

# You can also set truth using a dictionary, like below.
# If you do it this way, you do not need to
# set truth values for all parameters
c.configure_truth(color='w', ls=":", alpha=0.5) \
    .plot(filename="demoVarious6_TruthValues2.png",
          truth={"$x$": 0.0, "$y$": 5.0, r"$\beta$": 0.0})
