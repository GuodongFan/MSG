import numpy as np
import torch
from torch_geometric.data import Data

import torch.nn as nn
import torch

from torchtext.data import Field
import pandas as pd

from metric import getHR

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

m = nn.Conv1d(1, 4, 2)
input = torch.randn(8, 1, 50)
output = m(input)
print(output.shape)