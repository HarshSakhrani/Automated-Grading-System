from ocr import extract_text
from grader import result
import json
from pprint import pprint

def load_data(model_paper_json, student_paper_pdf):
	with open(model_paper_json, 'r') as j:
		model_paper = json.load(j)
	corpus = extract_text(student_paper_pdf)
	return model_paper, corpus

model_paper_json = './samples/history_model_paper.json'	
student_paper_pdf = './samples/history_answer_paper.pdf'
pretrained_file = '<path to>/bert-base-nli-mean-tokens'
# Download the pretrained model from: 
# https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/bert-base-nli-mean-tokens.zip

model_paper, corpus = load_data(model_paper_json, student_paper_pdf)
summary = result(model_paper, corpus, pretrained_file)

for item in summary:
	pprint(item)
