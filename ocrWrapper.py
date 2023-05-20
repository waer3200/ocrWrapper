import easyocr
import math
import pytesseract as tess
import keras_ocr
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import pandas as pd
import warnings
import collections
from multiprocessing import Process
from imutils import paths
import argparse
from collections import defaultdict
from collections import Counter

def writingLoop(i,resultEasy,resultPytesseract,resultKeras,resultPadle,resultTr):
   with open(f'arlogs/{i+1}.txt', "w", encoding="utf-8") as file:
     resultEasy = resultEasy.encode()
     resultKeras = resultKeras.encode()
     resultPytesseract = resultPytesseract.encode()
     resultPadle = resultPadle.encode()
     resultTr = resultTr.encode()
     file.write(f'EasyOcr:{resultEasy.decode()} length: {len(resultEasy.decode())} \n Pytess: {resultPytesseract.decode()} length: {len(resultPytesseract.decode())} \n keras: {resultKeras.decode()} length: {len(resultKeras.decode())} \n padle: {resultPadle.decode()} length: {len(resultPadle)} \n tr:{resultTr} length: {len(resultTr)} ')   

def cleaningChar(ocrResult):
   ocrResult = str(ocrResult)
   ocrResult = ocrResult.upper()
   charCleaned= ocrResult.replace('.','').replace('[','').replace(']','').replace("'","").replace('-','').replace(' ','').replace(',','').replace('|','').replace('\n','').replace('@','').replace('(','').replace(')','').replace('*','').replace(':','').replace('!','').replace('"','')
   return charCleaned
def laplaciancValue(image):
   return cv2.Laplacian(image, cv2.CV_64F).var()
def bluryDetection(path):
 ap = argparse.ArgumentParser()
 ap.add_argument("-t", "--threshold", type=float, default=100)
 args = vars(ap.parse_args())
 image = cv2.imread(path)
 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 fm = laplaciancValue(gray)
 text = "Not Blurry"
 if fm < args["threshold"]:
   text = "Blurry"
 return text
def funEasy(path , language) : 
    image = path
    image = cv2.imread(image)
    reader = easyocr.Reader([language], gpu=True , verbose=False)
    result = reader.readtext(image,detail=0, rotation_info=[90,180,270])
    cleanedResult = cleaningChar(result)
    return cleanedResult
def funPytesseract(path , language):
    if (language == 'en'):
       language = 'eng'
    image = path
    image = cv2.imread(image)
    tess.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe" #exe file for using ocr 
    result=tess.image_to_string(image,lang=language) #converts image characters to string
    cleanedResult = cleaningChar(result)
    return cleanedResult
def detect_w_keras(image_path):
	"""Function returns detected text from image"""

	# Initialize pipeline
	pipeline = keras_ocr.pipeline.Pipeline()

	# Read in image path
	read_image = keras_ocr.tools.read(image_path)

	# prediction_groups is a list of (word, box) tuples
	prediction_groups = pipeline.recognize([read_image]) 

	return prediction_groups[0]
def get_distance(predictions):
	# Point of origin
	x0, y0 = 0, 0 
	# Generate dictionary
	detections = []
	for group in predictions:
		top_left_x, top_left_y = group[1][0]
		bottom_right_x, bottom_right_y = group[1][1]
		center_x, center_y = (top_left_x + bottom_right_x)/2, (top_left_y + bottom_right_y)/2
		distance_from_origin = math.dist([x0,y0], [center_x, center_y])
		distance_y = center_y - y0
		detections.append({
							'text': group[0], 
							'center_x': center_x, 
							'center_y': center_y, 
							'distance_from_origin': distance_from_origin,
							'distance_y': distance_y
						})
	
	return detections 
def distinguish_rows(lst, thresh=15):
	sublists = []
	for i in range(0, len(lst)-1):
		if (lst[i+1]['distance_y'] - lst[i]['distance_y'] <= thresh):
			if lst[i] not in sublists:
				sublists.append(lst[i])
			sublists.append(lst[i+1])
		else:
			yield sublists
			sublists = [lst[i+1]]
	yield sublists

def funKeras(path , thresh , order='yes'):
	predictions = detect_w_keras(path)
	predictions = get_distance(predictions)
	predictions = list(distinguish_rows(predictions, thresh))
	predictions = list(filter(lambda x:x!=[], predictions))
	ordered_preds = []
	ylst = ['yes', 'y']
	for row in predictions:
		if order in ylst: row = sorted(row, key=lambda x:x['distance_from_origin'])
		for each in row: ordered_preds.append(each['text'])
	return ordered_preds
def funPadle(path , language):
   ocr = PaddleOCR(use_angle_cls=True, lang=language) 
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
def funTr(path):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    image = path
    image = cv2.imread(image)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_text = cleaningChar(generated_text)
    return generated_text
def election(easy,pytess,keras,padle,tr):
 chaines = [easy, pytess, keras, padle, tr]
 allResult = easy+pytess+keras+padle+tr
 #occurenceDic = dict(Counter(allResult))
 occurrenceDic = {}
 for char, count in Counter(allResult).items():
   occurrenceDic[char] = count
 occurenceList= list(occurrenceDic.keys())
 print(occurenceList)
 occurrences = defaultdict(list)
 coefficients = [1, 1, 3, 4, 5]
 finalDic={}
 for h , chaine in enumerate(chaines):
  print('*****************************************************')
  if(len(chaine)>0):
   hDic={}
   for i, char in enumerate(occurenceList):
     iDic= {}
     print(f'char: {char}')
     print(f'i{i}')
     positions=[]
     Jnotempty= []
     for j, stringTocompare in enumerate(chaines):
      if(len(stringTocompare)>0):
          Jnotempty.append(j)
          print(chaine)
          if(j== Jnotempty[0]):
           for k in range(len(chaine)):
            #print(chaine[k])
             if chaine[k] == char:
               positions.append(k)
          print(positions)
          dict = {value:[] for value in positions }
        
          for position in positions:  
            sumPositive=0
            sumNegative=0 
            if len(stringTocompare)>=0 and len(stringTocompare)>position and stringTocompare[position]==char:   
              print("Le caractère '{}' existe à la position {} dans la chaîne {}.".format(char, position,stringTocompare))
              if(position  in dict ):
               sumPositive=sumPositive+coefficients[j]
               dict[position] = sumPositive 
            else:
              print("Le caractère '{}' n'existe pas à la position {} dans la chaîne {}.".format(char, position,stringTocompare)) 
              if(position in dict):
               sumNegative= sumNegative-coefficients[j]
               dict[position]= sumNegative
          print("----------")
          for key in dict:
             if key in iDic:
                iDic[key] = dict[key] + iDic[key]  
          for key in dict:
             if key not in iDic:
                iDic[key] = dict[key]  
          for key in iDic:
             if key not in iDic:
                iDic[key] = iDic[key]
    
     print('----------------------------------------')        
     hDicFormat={}
     hDicFormat[char]=iDic    
     hDic.update(hDicFormat)
  #print('***********************************************')    
   if not finalDic:
     finalDic = hDic
   else:
     for key, value in hDic.items():
         if key in finalDic :
             finalDic[key].update(value)
         else:
             finalDic[key] = value    
 for key, nested_dict in finalDic.items():
     for nested_key, value in list(nested_dict.items()):
         if value < 0:
             del nested_dict[nested_key]
 finalDic = {key: nested_dict for key, nested_dict in finalDic.items() if nested_dict}

 for key, nested_dict in finalDic.items():
     remaining_keys = list(nested_dict.keys())
     nested_dict.clear()
     finalDic[key] = remaining_keys
 print(finalDic)
 max_position = max(max(positions) for positions in finalDic.values())
 table = [[] for _ in range(max_position + 1)]
 for key, positions in finalDic.items():
    for position in positions:
        table[position].append(key)

 table_string = ''
 for row in table:
    table_string += ' '.join(row) 

 print(table_string)
 return table_string

def electionAr(easy,pytess,padle):
   finalResult=[]
   easyL = len(easy)
   pytessL = len(pytess)
   padleL = len(padle)
   tableL = [easyL,pytessL,padleL]
   counter = collections.Counter(tableL)
   most_common = counter.most_common(1)[0]
   mostCommonLength = most_common[0]
   print(tableL)
   for i in range(mostCommonLength):
      votes = []
      coef = []  
      if(i <= padleL and i< mostCommonLength and padleL>0):
         v2= padle[i]
         votes.append(v2)
         coef.append(3)     
      if(i <= pytessL and i< mostCommonLength and pytessL>0):
         v4= pytess[i]
         votes.append(v4)
         coef.append(1)      
      if(i<easyL and i< mostCommonLength):
         v5= easy[i]
         votes.append(v5)
         coef.append(2)
      print(votes)
      votesCount={}
      for i, item in enumerate(votes):
        if item in votesCount:
          votesCount[item]["count"] += 1
          votesCount[item]["positions"].append(i)
        else:
          votesCount[item] = {"count": 1, "positions": [i]}
      singleCharVote = []
      characters= []
      for item, count_dict in votesCount.items():
         print(f"{item} appears {count_dict['count']} times at positions {count_dict['positions']}")
         itemVote=0
         pos=count_dict['positions']
         for i in pos:
            itemVote = itemVote+coef[i]
         print(f"{item} has {itemVote} votes")
         characters.append(item)
         singleCharVote.append(itemVote)
      maxIndex = singleCharVote.index(max(singleCharVote))
      chosenOne = characters[maxIndex]
      print(f"{characters[maxIndex]} get biggest number of votes witch is {max(singleCharVote)}")
      finalResult.append(chosenOne)
      print(f'coefficient {coef}')
      print('----------------------------------------------------------------------------')
   print(f'The result of the pull is {finalResult}')
   return finalResult
def testIfAlgeria(license):
   licenseLength = len(license)
   has_alphabetic = False
   for string in license:
     if string.isalpha():
        has_alphabetic = True
        break
   if((licenseLength >=8 and licenseLength <=11  )  and has_alphabetic == False):
      return True
   else : 
      return False
def testIfEu(license):
   licenseLength = len(license)
   has_alphabetic = False
   for string in license:
     if string.isalpha():
        has_alphabetic = True
        break
   if(licenseLength==7 and has_alphabetic == True):
      return True 
def identity(license):
 wilayas = {
    "01": "Adrar",
    "02": "Chlef",
    "03": "Laghouat",
    "04": "Oum El Bouaghi",
    "05": "Batna",
    "06": "Béjaïa",
    "07": "Biskra",
    "08": "Béchar",
    "09": "Blida",
    "10": "Bouira",
    "11": "Tamanrasset",
    "12": "Tébessa",
    "13": "Tlemcen",
    "14": "Tiaret",
    "15": "Tizi Ouzou",
    "16": "Alger",
    "17": "Djelfa",
    "18": "Jijel",
    "19": "Sétif",
    "20": "Saïda",
    "21": "Skikda",
    "22": "Sidi Bel Abbès",
    "23": "Annaba",
    "24": "Guelma",
    "25": "Constantine",
    "26": "Médéa",
    "27": "Mostaganem",
    "28": "M'Sila",
    "29": "Mascara",
    "30": "Ouargla",
    "31": "Oran",
    "32": "El Bayadh",
    "33": "Illizi",
    "34": "Bordj Bou Arréridj",
    "35": "Boumerdès",
    "36": "El Tarf",
    "37": "Tindouf",
    "38": "Tissemsilt",
    "39": "El Oued",
    "40": "Khenchela",
    "41": "Souk Ahras",
    "42": "Tipaza",
    "43": "Mila",
    "44": "Aïn Defla",
    "45": "Naâma",
    "46": "Aïn Témouchent",
    "47": "Ghardaïa",
    "48": "Relizane",
    "49":"El M'Ghair",
    "50" : "El Menia" ,
    "51" :"Ouled Djellal" ,
    "52" : "Bordj Baji Mokhtar",
    "53" :  "Béni Abbès",
    "54": "Timimoun" ,
    "55": "Touggourt" ,
    "56" : "Djanet",
    "57" : "In salah",
    "58" :  "In Guezzam",
   }
 vehiculeTypeList = {"1" : "Véhicule privée",
                 "2" : "Camion",
                 "3" : "Camionettes",
                 "4" : "Buses",
                 "5" : "Tracteur routier",
                 "6" : "Type de tracteur autre",
                 "7" : "Véhicule spéciale",
                 "8" : "Remorque",
                 "9" : "Moto"
                 }
 license = list(reversed(license))
 codeWilaya = license[1]+ license[0]
 wilayaName = wilayas.get(codeWilaya)
 year = license[3] + license[2]
 yearInt = int(year)
 yearChar = str(yearInt)
 completeYear=""
 if( yearInt >= 0 and yearInt<=23):
    if(len(yearChar)== 1):
      completeYear = "200"+yearChar
    if(len(yearChar)== 2):
      completeYear = "20"+yearChar
 if(yearInt >= 70 and yearInt <=99):
    yearChar = str(yearInt)
    completeYear = "19"+yearChar   
 print(completeYear)
 vehiculeTypeNumber = license[4]
 vehiculeTypeName = vehiculeTypeList.get(vehiculeTypeNumber)
 print(vehiculeTypeName)
 vehiculeInformation = {"year": completeYear,"pays": "algerie", "wilaya" : wilayaName , 'vehiculeType' : vehiculeTypeName}
 return vehiculeInformation
def main():
  #for i in range(68):
   language = 'en' 
   path = f'C:/Users/walid/Desktop/Pfe/ocrWrapper/Images/lp66.png'
   blury = bluryDetection(path)
   print(blury)
   resultEasy= funEasy(path , language)
   resultPytesseract = funPytesseract(path, language)
   resultKeras = cleaningChar(funKeras(path,15,'yes'))
   resultPadle = funPadle(path, language)
   resultTr = funTr(path)
   print('----------------------------------------------------------------------------')
   #print(f'For the image number whos number is  {i+1} the results are ') 
   print('--------------------------------Easyocr-------------------------------------')
   print(resultEasy)
   print(f'Taille de la chaine : {len(resultEasy)}')
   print('--------------------------------pytess--------------------------------------')
   print(resultPytesseract)
   print(f'Taille de la chaine: {len(resultPytesseract)}')
   print('--------------------------------keras-------------------------------------')
   print(resultKeras)
   print(f'Taille de la chaine: {len(resultKeras)}')
   print('--------------------------------padle--------------------------------------')
   print(resultPadle)
   print(f'Taille de la chaine: {len(resultPadle)}')
   print('--------------------------------TR--------------------------------------')
   print(resultTr)
   print(f'Taille de la chaine: {len(resultTr)}')
   print('----------------------------------------------------------------------------')
   
   #writingLoop(i,iresultEasy,resultPytesseract,resultKeras,resultPadle,resultTr)
   poolingResult = election(resultEasy,resultPytesseract,resultKeras,resultPadle,resultTr)
   algeria= testIfAlgeria(poolingResult)
   europeanUnion = testIfEu(poolingResult)
   if(algeria == True):
     information=identity(poolingResult)
     print(information)
   if(europeanUnion == True):
      print("EU")
if __name__ == "__main__":
    main()