import easyocr
import pytesseract as tess
import keras_ocr
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def cleaningChar(ocrResult):
   ocrResult = str(ocrResult)
   charCleaned= ocrResult.replace('.','').replace('[','').replace(']','').replace("'","").replace('-','').replace(' ','')
   return charCleaned

def funEasyOcr(path) : 
    image = path
    image = cv2.imread(image)
    reader = easyocr.Reader(['en'], gpu=True , verbose=False)
    result = reader.readtext(image,detail=0, rotation_info=[90,180,270])
    cleanedResult = cleaningChar(result)
    return cleanedResult

def funPytesseract(path):
    image = path
    image = cv2.imread(image)
    tess.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe" #exe file for using ocr 
    result=tess.image_to_string(image,lang='eng') #converts image characters to string
    cleanedResult = cleaningChar(result)
    return cleanedResult

def funKerasOcr(path):
    images = [
    keras_ocr.tools.read(img) for img in [path]]
    pipeline = keras_ocr.pipeline.Pipeline()
    result = pipeline.recognize(images)
    result = pd.DataFrame(result[0])
    size = len(result) 
    list=[]
    for i in range(size) :
       list.append(result[0][i])

    result = ''.join(list)
    cleanedResult = cleaningChar(result)
    return cleanedResult

def funPadleOcr(path):
   ocr = PaddleOCR(use_angle_cls=True, lang='en') 
   result = ocr.ocr(path, cls=True)
   for idx in range(len(result)):
     res = result[idx]
     #for line in res:
        #print(line) 
   result = result[0]
   txts = [line[1][0] for line in result]
   print('-----------------------txts')
   txtsCleaned = cleaningChar(txts) 
   return txtsCleaned


def funTrOcr(path):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    image = path
    image = cv2.imread(image)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


path = 'Images/lp81.png'
resultEasyocr= funPytesseract(path)
print(resultEasyocr)
   
    
 

  
    


