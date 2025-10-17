"""EMFieldML: Electromagnetic Field Machine Learning package."""

# Direct imports with actual class names - no aliases for transparency
from EMFieldML.ActiveLearning.ActiveLearning import ActiveLearning
from EMFieldML.ActiveLearning.Decide6points import Decide6points
from EMFieldML.config import config, get_logger, paths, template
from EMFieldML.FEKO.FEKORunner import FekoRunner
from EMFieldML.Learning.Learning import FieldGPRegressor
from EMFieldML.Learning.YDataMaker import TargetDataBuilder
from EMFieldML.Modeling.PolyCubeMaker import PolyCubeMaker
from EMFieldML.Visualize import Prediction
from EMFieldML.Visualize.Visualize import Visualize

# Results classes
try:
    from EMFieldML.Results.Graph import GraphMaker
    from EMFieldML.Results.MagnitudeMapMaker import MakeYData
    from EMFieldML.Results.Results import ResultMaker
    from EMFieldML.Results.TestMaker import TestMaker
except ImportError:
    # These might have import issues, so make them optional
    GraphMaker = None
    MakeYData = None
    ResultMaker = None
    TestMaker = None

__all__ = [
    # Active learning
    "ActiveLearning",
    "Decide6points",
    # FEKO runner
    "FekoRunner",
    # Learning - actual class names
    "FieldGPRegressor",
    "TargetDataBuilder",
    # Modeling
    "PolyCubeMaker",
    # Visualization
    "Visualize",
    "Prediction",
    # Results
    "GraphMaker",
    "MakeYData",
    "ResultMaker",
    "TestMaker",
    # Config
    "config",
    "paths",
    "template",
    "get_logger",
]
