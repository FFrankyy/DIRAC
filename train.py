from SGRL import SGRL
import networkx as nx
import numpy as np
from tqdm import tqdm
import time
from config import parsers

args = parsers()

def Train():
    sg = SGRL()
    sg.Train()


if __name__=="__main__":
    Train()
