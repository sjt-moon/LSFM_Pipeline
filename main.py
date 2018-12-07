from pipeline import Pipeline
from helper import loader
import os
import warnings
warnings.filterwarnings("ignore")

p = Pipeline(base_model_path='../facegen/1/1_0001.obj')
lsfm, logs = p.run(input_path='../facegen/')

loader.save(lsfm, os.path.join(p.output_path, "lsfm"))
loader.save(logs, os.path.join(p.output_path, "lsfm-training_logs"))
