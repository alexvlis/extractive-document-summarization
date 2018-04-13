''' 
Usage:
    from rouge import Rouge
    r = Rouge()
    s = r.saliency(reference, system)    
'''
from pyrouge import Rouge155
import numpy as np

class Rouge():
    def __init__(self, alpha=0.5):
        self.r = Rouge155()

        self.r.model_dir = 'model_summaries'
        self.r.system_dir = 'system_summaries'
        self.r.system_filename_pattern = 'text.(\d+).txt'
        self.r.model_filename_pattern = 'text.[A-Z].#ID#.txt'
     
        self.alpha = alpha

    def saliency(self, reference=None, system=None):
        if reference is not None:
            np.savetxt("model_summaries/text.A.001.txt", reference, newline="\n", fmt="%s")
        
        if system is not None:
            np.savetxt("system_summaries/text.001.txt", system, newline="\n", fmt="%s")

        output = self.r.convert_and_evaluate()
        output = self.r.output_to_dict(output)
        R1 = output['rouge_1_f_score']
        R2 = output['rouge_2_f_score']

        return self.alpha * R1 + (1 - self.alpha) * R2

if __name__ == "__main__":
    ref = np.array(["blah blah blah", "motherfucker"])
    model = np.array(["motherfucker", "blah"])

    r = Rouge()
    print(r.saliency(ref, model))
