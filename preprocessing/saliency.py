from pyrouge import Rouge155

def main():
    r = Rouge155()

    r.model_dir = 'model_summaries'
    r.system_dir = 'system_summaries'
    r.system_filename_pattern = 'text.(\d+).txt'
    r.model_filename_pattern = 'text.[A-Z].#ID#.txt'
    
    output = r.convert_and_evaluate()
    output = r.output_to_dict(output)
    print("ROUGE-1 score:", output['rouge_1_f_score'])
    print("ROUGE-2 score:", output['rouge_2_f_score'])

if __name__ == "__main__":
    main()
