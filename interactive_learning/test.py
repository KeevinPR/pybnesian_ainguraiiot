#!/usr/bin/env python
# coding: utf-8

# In[21]:
import os
os.environ["OMP_NUM_THREADS"] = '2'


import importlib
import pybnesian as pbn

import numpy as np
import os
import json
import shutil

from dash import Dash, jupyter_dash, html, Input, Output, State, callback, callback_context, dcc, no_update
from dash_extensions.enrich import DashProxy, LogTransform, DashLogger
from flask import session
from flask_session import Session
from flask_caching import Cache  # Import Cache for caching
# import redis
import dash_ag_grid as dag
import dash_cytoscape as cyto
import networkx as nx
import pandas as pd

import importlib
import frontend
import callbacks
importlib.reload(frontend)
importlib.reload(callbacks)
import pickle
from frontend import *


app.run(debug=True, use_reloader=False, port=8050)
