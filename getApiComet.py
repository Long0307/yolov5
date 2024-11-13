## import comet_ml at the top of your file
from comet_ml import Experiment

## Create an experiment with your api key
experiment = Experiment(
    api_key="Your API Key",
    project_name="Your Project Name",
    workspace="Your Workspace Name",
)