''' 
Usage:
    from rouge import Rouge
    r = Rouge()
    s = r.saliency(reference, system)    
'''
from pyrouge import Rouge155
import numpy as np

class Rouge():
    def __init__(self):
        pass

    def saliency(self, reference=None, system=None, alpha=0.5):
        self.r = Rouge155()

        self.r.model_dir = 'model_summaries'
        self.r.system_dir = 'system_summaries'
        self.r.system_filename_pattern = 'text.(\d+).txt'
        self.r.model_filename_pattern = 'text.[A-Z].#ID#.txt'

        open('model_summaries/text.A.001.txt', 'w').close()
        open('system_summaries/text.001.txt', 'w').close()
        if reference is not None:
            np.savetxt("model_summaries/text.A.001.txt", reference, newline="\n", fmt="%s")
        
        if system is not None:
            np.savetxt("system_summaries/text.001.txt", system, newline="\n", fmt="%s")

        output = self.r.convert_and_evaluate()
        output = self.r.output_to_dict(output)
        R1 = output['rouge_1_f_score']
        R2 = output['rouge_2_f_score']

        return alpha * R1 + (1 - alpha) * R2

if __name__ == "__main__":
    ref = np.array([" The territory's stock market opened this morning sharply lower"])
    model = np.array(["      Deteriorating relations between Britain and China over Hong Kong's political future has cast a pall over the colony's financial markets",
         ' Governor Patten has proposed elections in 1994 and 1995 to allow for greater democracy for the people and is supported by the people',
          ' Patten has repeatedly indicated a go-it-alone possibility marking a change in British policy toward Hong Kong and China, which had placed priority on reaching agreement with Beijing',
           '  China claims that financing proposals from the Hong Kong government would put a heavy financial burden on the post-1997 community',
            '  China could sacrifice the economic well-being of its people in response to a perceived threat to its authority',
             ' '])

    r = Rouge()
    print(r.saliency(ref, model, alpha=1.0))
    print(r.saliency(ref, model))
