from google.cloud import vision
import os
from pdf2image import convert_from_path
import json
import io

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
client = vision.ImageAnnotatorClient()

def assemble_word(word):
	assembled_word=""
	for symbol in word.symbols:
	    assembled_word+=symbol.text
	return assembled_word


def extract_text(pdf_file):
	images = convert_from_path(pdf_file)

	for image in images:
		byte_array = io.BytesIO()
		image.save(byte_array, format='JPEG')
		content = byte_array.getvalue()

		img = vision.types.Image(content=content)
		response = client.document_text_detection(image=img)
		document = response.full_text_annotation

		corpus = []
		for page in document.pages:
			for block in page.blocks:
				text = []
				for paragraph in block.paragraphs:
					for word in paragraph.words:
						assembled_word = assemble_word(word)
						text.append(assembled_word)
				text_str = ' '.join(text)
				text_str = text_str.replace('Scanned with CamScanner', '')
				if len(text_str) > 2:
					corpus.append(text_str)
	return corpus


