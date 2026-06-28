import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # добавляем корень

from plugins.data import get_data
from plugins.evaluate import evaluate_model
from plugins.fit import fit_model

if __name__ == "__main__":
    get_data()
    fit_model()
    evaluate_model()


