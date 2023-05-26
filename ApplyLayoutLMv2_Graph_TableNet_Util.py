# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:05:24 2022

@author: ngoclan
"""


import os
import re
import difflib
import pdfplumber #PyPDF2
import tabula
#os.system('pip install git+https://github.com/huggingface/transformers.git --upgrade')
#os.system('pip install pyyaml==5.1')
# workaround: install old version of pytorch since detectron2 hasn't released packages for pytorch 1.9 (issue: https://github.com/facebookresearch/detectron2/issues/3158)
#os.system('pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html')

# install detectron2 that matches pytorch 1.8
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
#os.system('pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html')

## install PyTesseract
#os.system('pip install -q pytesseract')


import cv2
import numpy as np
import pandas as pd
from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2TokenizerFast, LayoutLMv2Tokenizer, LayoutLMv2ForTokenClassification
from PIL import Image, ImageDraw, ImageFont, ImageOps
import datefinder
from dateutil.parser import parse
from price_parser import Price


# define id2label, label2color
#labels = dataset.features['ner_tags'].feature.names
labels = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
id2label = {v: k for v, k in enumerate(labels)}
label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def iob_to_label(label):
    label = label[2:]
    if not label:
      return 'other'
    return label

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    color_converted = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return Image.fromarray(color_converted)
    #return Image.fromarray(img)

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #return np.asarray(img)

def shear(img, shear):
    shear = img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))
    return shear

def remove_black_dot(file):
    img = cv2.imread(file, 0)
    _, blackAndWhite = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, 4, cv2.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if sizes[i] >= 7: #8:   #filter small dotted regions
            img2[labels == i + 1] = 255

    res = cv2.bitwise_not(img2)
    cv2.imwrite(file, res)
    
def remove_black_dot2(file):
     img = cv2.imread(file)   
     # denoising of image saving it into dst image 
     dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
     cv2.imwrite(file, dst)
     
def remove_black_dot3(file):         
    im = cv2.imread(file)
    gray = 255 - cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # prepare a mask using Otsu threshold, then copy from original. this removes some noise
    __, bw = cv2.threshold(cv2.dilate(gray, None), 128, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
    gray = cv2.bitwise_and(gray, bw)
    # make copy of the low-noise underlined image
    grayu = gray.copy()
    imcpy = im.copy()
    # scan each row and remove lines
    for row in range(gray.shape[0]):
        avg = np.average(gray[row, :] > 16)
        if avg > 0.9:
            cv2.line(im, (0, row), (gray.shape[1]-1, row), (0, 0, 255))
            cv2.line(gray, (0, row), (gray.shape[1]-1, row), (0, 0, 0), 1)
    
    cont = gray.copy()
    graycpy = gray.copy()
    # after contour processing, the residual will contain small contours
    residual = gray.copy()
    # find contours
    _, contours, hierarchy = cv2.findContours(cont, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        # find the boundingbox of the contour
        x, y, w, h = cv2.boundingRect(contours[i])
        if 10 < h:
            cv2.drawContours(im, contours, i, (0, 255, 0), -1)
            # if boundingbox height is higher than threshold, remove the contour from residual image
            cv2.drawContours(residual, contours, i, (0, 0, 0), -1)
        else:
            cv2.drawContours(im, contours, i, (255, 0, 0), -1)
            # if boundingbox height is less than or equal to threshold, remove the contour gray image
            cv2.drawContours(gray, contours, i, (0, 0, 0), -1)
    
    # now the residual only contains small contours. open it to remove thin lines
    st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, st, iterations=1)
    # prepare a mask for residual components
    __, residual = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY)
    
    #cv2.imshow("gray", gray)
    #cv2.imshow("residual", residual)   
    
    # combine the residuals. we still need to link the residuals
    combined = cv2.bitwise_or(cv2.bitwise_and(graycpy, residual), gray)
    # link the residuals
    st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 7))
    linked = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, st, iterations=1)
    #cv2.imshow("linked", linked)
    # prepare a msak from linked image
    __, mask = cv2.threshold(linked, 0, 255, cv2.THRESH_BINARY)
    # copy region from low-noise underlined image
    clean = 255 - cv2.bitwise_and(grayu, mask)
    #cv2.imshow("clean", clean)
    #cv2.imshow("im", im)
    cv2.imwrite(file, clean)
    
def checkNanValue (value):
    try:
        return np.isnan(value)
    except:
        return False
     
def process_image(feature_extractor, tokenizer, model, image, lstHasToBeQuestion=None, lstHasToBeAnswer=None, lstWordsFromPdf=None):
    width, height = image.size
    
    # get words, boxes
    encoding_feature_extractor = feature_extractor(image, return_tensors="pt")
    words, boxes = encoding_feature_extractor.words, encoding_feature_extractor.boxes

    # encode
    encoding = tokenizer(words, boxes=boxes, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    offset_mapping = encoding.pop('offset_mapping')
    encoding["image"] = encoding_feature_extractor.pixel_values

    # forward pass
    outputs = model(**encoding)

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]

  
    origBoxes = []
    for box in boxes[0] : 
        origBoxes.append(unnormalize_box(box, width, height))
    
    pdWordsBoxes = pd.concat([pd.DataFrame({'words': words[0]}), pd.DataFrame({'origBoxes': origBoxes})], axis=1, join='inner')
    pdWordsBoxes['origBoxes_STR'] = pdWordsBoxes['origBoxes'].astype(str)
    pdPred = pd.DataFrame(zip(true_boxes, true_predictions),columns = ['true_boxes', 'true_predictions'])
    pdPred['true_boxes_STR'] = pdPred['true_boxes'].astype(str)
    finalPD = pd.merge(pdWordsBoxes, pdPred, how='left', left_on = 'origBoxes_STR', right_on = 'true_boxes_STR')
    finalPD = finalPD.dropna(subset = ['true_predictions'])
    #remove all linhtinh characters
    finalPD = finalPD[~finalPD.words.isin(["@", ",", ".", ":","-","=","+", "|","TT","II","||","TTT","III","|||","IIL","MMM","MMA"]) ] #"&",
    finalPD = finalPD[~finalPD['words'].str.contains("TTT|II|LLL|MMM|MMA") ] 
    
    #HoaDon tiếng Việt thì 3 chữ, tiếng Anh thì 8 chữ
    minLengText2Check =  3 if ('HÓA' in lstWordsFromPdf or 'ĐƠN' in lstWordsFromPdf) else 6
    for index, row in finalPD.iterrows():
        word = row['words']
        ##This is for correct text
        if lstWordsFromPdf and len(word) >= minLengText2Check:
            closestWords = difflib.get_close_matches(word, lstWordsFromPdf)
            if closestWords and len(word) - len(closestWords[0]) >=-4 and len(word) - len(closestWords[0]) <= 3: 
                word = closestWords[0]
                finalPD.at[index, 'words'] = word
            else:
                #finalPD.at[index, 'true_predictions'] = "O"
                continue
        try:
            if row['origBoxes'][1] <= height/2 and row['true_predictions'] == "O":
                finalPD.at[index, 'true_predictions'] = "I-ANSWER" 
            is_date = False
            for match in datefinder.find_dates( word):
                if len(word) >= 6 and match.year >= 2022:
                    is_date = True
                    word = closestWords[0]
                    finalPD.at[index, 'words'] = word
            if is_date :
                closestWords = difflib.get_close_matches(word, lstWordsFromPdf)
                if 'ANSWER' not in row['true_predictions']:
                    finalPD.at[index, 'true_predictions'] = "I-ANSWER"
            if lstHasToBeQuestion and any(q in word for q in lstHasToBeQuestion) and 'QUESTION' not in row['true_predictions']:
                finalPD.at[index, 'true_predictions'] = "I-QUESTION"
            if lstHasToBeAnswer and any(re.search(a, word) for a in lstHasToBeAnswer) and 'ANSWER' not in row['true_predictions']:
                finalPD.at[index, 'true_predictions'] = "I-ANSWER"                
        except Exception as e:
            print(e)
            pass
    
    try:#maybe TOTAL not in the image
        finalPD['y1'] = [x[3] for x in finalPD['origBoxes']]
        filter1 = finalPD['words'].str.contains('TOTAL')
        filter2 = finalPD['y1'] > 1000 #y1 > 1000
        filter3 = finalPD['words'].str.contains('Container')
        if height > 1550:
            y1Df = finalPD.where(filter1 & filter2).dropna().reset_index(drop=True)['y1']
            if y1Df.shape[0] == 1 or y1Df.shape[0] == 2:
                y1_of_total = y1Df[0]
            elif y1Df.shape[0] == 3: #img has "FOB TOTAL", "CFR TOTAL" and "TOTAL"
                y1_of_total = y1Df[1]
            else:
                y1_of_total = y1Df[0]
            finalPD.drop(finalPD[(finalPD.y1 > y1_of_total) & (finalPD.y1 < y1_of_total+50)].index, inplace=True) #drop linh tinh character below TOTAL like //////////////////////
        #TODELETE 20220802
        else:#case process for bottom half
             #this image is halfBottom --> only neeed to get info Quantity, Gross-weight, Net-weight, Amount
            try: 
                y1_of_total = finalPD.where(filter1).dropna().reset_index(drop=True)['y1'][-1]
            except Exception as e:
                print("No y1_of_total")
                y1_of_total = 2222
            try:
                y1_of_container = finalPD.where(filter3).dropna().reset_index(drop=True)['y1'][0]-100 #tru hao
            except Exception as e:
                print("No y1_of_container")
                y1_of_container = 2222
            y1 = y1_of_total #if y1_of_total<y1_of_container else y1_of_container
            finalPD.drop(finalPD[(finalPD.y1 > y1+100)].index, inplace=True) 
            
        finalPD.reset_index(drop=True, inplace=True)
    except Exception as e:
        print(e)
        finalPD.reset_index(drop=True, inplace=True)
        pass        
    
    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(finalPD['true_predictions'], finalPD['true_boxes']):
        #print(prediction)
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color[predicted_label], font=font)

    return image, finalPD


import glob
def do_extract_from_png(image, file, feature_extractor, tokenizer, model, graph, lstHasToBeQuestion=None, lstHasToBeAnswer=None, extQuestionAnswerGraphFunc=None, lstWordsFromPdf=None):    
    width, height = image.size
    #===================== Use ML to classify word ========================
    image, table = process_image(feature_extractor, tokenizer, model, image, lstHasToBeQuestion, lstHasToBeAnswer, lstWordsFromPdf)
    image.save(file+'.jpg')
    #================ Use Graph to math Question-Answer ===================
    tx = graph.begin()
    tx.evaluate('''  MATCH (n) DETACH DELETE n  ''')
    graph.commit(tx)
    #CREATE NODES
    tx = graph.begin()
    #ttt.append(table)
    
    for index, row in table.iterrows():
        try:
            if row['true_predictions'] == 'O':
                tx.evaluate('''   //sometime $x1 is too short (not correct) --> heuristic adjsutment
                   MERGE (o:OTHER  {word:$label2, x0:$x0, y0:$y0, x1:(case when $x1 > $x0+10*size($label2) Then $x1 else $x0+10*size($label2) end), y1:$y1})
                   ''', parameters = {'label2': row['words'], 'x0':row['origBoxes'][0], 'y0':row['origBoxes'][1]
                                                           , 'x1':row['origBoxes'][2], 'y1':row['origBoxes'][3]})
            elif 'QUESTION' in row['true_predictions']:
                tx.evaluate('''
                   MERGE (q:QUESTION {word:$label2, x0:$x0, y0:$y0,  x1:(case when ($x1 > $x0+10*size($label2) and $x1-$x0 < 2*10*size($label2)) Then $x1 else $x0+10*size($label2) end), y1:$y1})
                   ''', parameters = {'label2': row['words'], 'x0':row['origBoxes'][0], 'y0':row['origBoxes'][1]
                                                           , 'x1':row['origBoxes'][2], 'y1':row['origBoxes'][3]})
            elif 'ANSWER' in row['true_predictions']:
                tx.evaluate('''
                   MERGE (a:ANSWER {word:$label2, x0:$x0, y0:$y0,  x1:(case when ($x1 > $x0+10*size($label2) and $x1-$x0 < 2*10*size($label2)) Then $x1 else $x0+10*size($label2) end), y1:$y1})
                   ''', parameters = {'label2': row['words'], 'x0':row['origBoxes'][0], 'y0':row['origBoxes'][1]
                                                           , 'x1':row['origBoxes'][2], 'y1':row['origBoxes'][3]})
        except:
            print(row['true_predictions'])
        
    graph.commit(tx)
    
    #CREATE RELATIONSHIPs
    tx = graph.begin()
    tx.evaluate('''
                match (a:QUESTION), (b:QUESTION)
                where a.x0 - b.x0 < 0 and  a.x1 - b.x0 > -25
                     and abs(a.y0 - b.y0) < 10 
                     and not(a.word contains 'Gross-weight' or a.word contains 'Gross-w') //sometime Gross-weight is too near with Measure so it create relation
                create (a)-[:NEXTWORD {x:abs(a.x1-b.x0)}]->(b);
                ''')
                
    tx.evaluate('''
                match (a:ANSWER), (b:ANSWER)
                where (( a.x1-a.x0 >= 45 and  //date modify: 20220725
                         b.x0 -(a.x0+a.x1)/2 > 0 
                         and  b.x0 -(a.x0+a.x1)/2 < (case when a.y0>1100 and a.x0>$width/2 
                                                   then 90 
                                                   else (case when a.x1-a.x0 > 20 Then a.x1-a.x0 else 20 end)
                                                   end) ) //apoc.coll.max([a.x1-a.x0, 20]) //some rectangle not correctly bound the word --> heuristic
                      or (a.x1-a.x0 < 45 and -2 < b.x0-a.x1 < 45 )  //date modify: 20220725
                      )
                     and abs(a.y0 - b.y0) < 18
                create (a)-[:NEXTWORD {x:abs(a.x1-b.x0), y:abs(a.y0 - b.y0)}]->(b);
                ''', parameters = {'width': width})
                           
    tx.evaluate('''
                match (q:QUESTION), (a:ANSWER)
                where NOT (:ANSWER)-[:NEXTWORD]->(a)
                    and q.x1 < a.x0 and   a.x0 - q.x1 < 450
                    //and q.x1 < a.x0
                    and abs((q.y0+q.y1)/2 - (a.y0+a.y1)/2) < 10  //some rectangel is tall --> calculate average rectangle heigh (y)
                create (q)-[:RELATED {y:abs((q.y0+q.y1)/2 - (a.y0+a.y1)/2), x:(a.x0 - q.x1)}]->(a);
                ''')
    graph.commit(tx)
    
    #DELETE Redundant relations
    tx = graph.begin()
    tx.evaluate('''
                match (a:QUESTION)-[r1:NEXTWORD]->(b:QUESTION),
                    (a:QUESTION)-[r2:NEXTWORD]->(c:QUESTION)
                where r2.x > r1.x
                delete r2;
                ''')
                
    tx.evaluate('''
                with ['description', 'quantity', 'unit', 'gross', 'net', 'amount'] as quest 
                match (a:QUESTION)-[r:NEXTWORD]->(b:QUESTION)
                where any (word in quest where toLower(b.word) contains word)
                delete r;
                ''')
                
    tx.evaluate('''
                match (a:ANSWER)-[r1:NEXTWORD]->(b:ANSWER),
                    (a:ANSWER)-[r2:NEXTWORD]->(c:ANSWER)
                where (r2.x > r1.x  or r2.y - r1.y  > 15)
                delete r2;
                ''')
    graph.commit(tx)

    
    ##################### extend Question-->Answer Graph Func #####################
    if extQuestionAnswerGraphFunc:
        extQuestionAnswerGraphFunc(graph, file=file,tablePD=table, considerWidth=width)
    
    ################################   #GET RESULT  ############################
    query ='''
                match p=(q:QUESTION) - [*1..] -> (a:ANSWER)
                where any(rel in relationships(p) WHERE type(rel) = "RELATED")
                    and not((:QUESTION)-[:NEXTWORD]->(q:QUESTION)) //get the longest path
                    and not((a:ANSWER)-[:NEXTWORD]->()) //get the longest path
                return  trim(reduce(t="", str in [n in nodes(p) where 'QUESTION' in labels(n) | n.word] | t+str+" ")) as question,
                        trim(reduce(t="", str in [n in nodes(p) where 'ANSWER'   in labels(n) | n.word] | t+str+" ")) as answer
                order by a.y0 //order by from upper to lower text
                //order by length(p) DESC
                //limit 1
                ;   
           '''
    return graph.run(query).to_data_frame().drop_duplicates()

switcherExtractInvoice = {
        1: "Invoice No",
        2: "Invoice Date",
        3: "Arrival",
        4: "Departure",
        5: "Carrier",
        6: "Sailing on About",
        7: "Notify Party",
        8: "Shipper",
        9: "Amount",
        10: "UnitPrice",
        11: "G.W",
        12: "N.W",
        13: "Currency",
        14: "Container \ Seal No",
        15: "Description of Goods",
        16: "UNIT",
        17: "PackingMethod"
}
def findMissingData(arr, switcher = switcherExtractInvoice):
    missing_elements = []
    for ele in range(1,len(switcher)+1):
        if ele not in arr:
            missing_elements.append(ele)
    return missing_elements

def numbers_to_header(argument, switcher = switcherExtractInvoice):
    return switcher.get(argument, "nothing")

def getContainerSealNo_fromPdf(pdfFile):
    try:
        strContainerSealNo = ""
        dfs = tabula.read_pdf(pdfFile, pages='all')    
        tableContainerSealNo = dfs[0]
        tableContainerSealNo = tableContainerSealNo.T.reset_index().T                
        if tableContainerSealNo.shape[0] < 10 and 7 < len(tableContainerSealNo.iloc[0][0]) < 15 and len(tableContainerSealNo.iloc[0][1]) < 15:
            #format Container | Seal | Lbs | Roll | Amount  --> 5 columns
            tableContainerSealNo[tableContainerSealNo.columns[0]] = tableContainerSealNo[tableContainerSealNo.columns[0]].astype(str)
            tableContainerSealNo[tableContainerSealNo.columns[1]] = tableContainerSealNo[tableContainerSealNo.columns[1]].astype(str)
            if len(tableContainerSealNo.columns) == 5:
                
                tableContainerSealNo[r'Container \ Seal No'] = tableContainerSealNo[tableContainerSealNo.columns[0]] + " / " + tableContainerSealNo[tableContainerSealNo.columns[1]]
                strContainerSealNo = '\n'.join(tableContainerSealNo[~tableContainerSealNo[r'Container \ Seal No'].isnull()][r'Container \ Seal No']) 
                #====================== 6 columns ============================
                #    CONTAINER NO.						
                #   WHSU5799895 	/ WHLR312145		NET W'T	     18ROLL(S)	    18,424 KGS	    40,617 LBS
                #  				                        GROSS W'T		            18,928 KGS	    41,728 LBS
                #   WHSU5780483 	/ WHLR312052		NET W'T	     19ROLL(S)	    19,435 KGS	    42,846 LBS
                #  				                        GROSS W'T		            19,965 KGS	    44,015 LBS
                #   WHSU5263120 	/ WHLR312003		NET W'T	     18ROLL(S)	    18,465 KGS	    40,708 LBS
                #  				                        GROSS W'T		            18,968 KGS      41,817 LBS 
                #==================================================
            elif len(tableContainerSealNo.columns) == 6:
                tableContainerSealNo[r'Container \ Seal No'] = tableContainerSealNo[tableContainerSealNo.columns[0]] + " " + tableContainerSealNo[tableContainerSealNo.columns[1]]
                strContainerSealNo = '\n'.join(tableContainerSealNo[~tableContainerSealNo[r'Container \ Seal No'].isnull()][r'Container \ Seal No']) 
        
        if "Unname" in strContainerSealNo or "TOTAL" in strContainerSealNo:
            return ""
        
        return strContainerSealNo 
    except:
        return ""

def invoice_info_extract_from_png(feature_extractor, tokenizer, model, graph, path, question2header_func, switcher = switcherExtractInvoice, tablePred=None, out_collection=None, cut_image = "left", image_shear=0.03, 
                                  lstHasToBeQuestion=None, lstHasToBeAnswer=None, extQuestionAnswerGraphFunc=None):
    reverseSwitcher = dict(zip(switcher.values(), switcher.keys()))
    fileFullnames = glob.glob(path + "\*.png")
    for file in fileFullnames:
        pdfFile = file[0:-4]
        lstWordsFromPdf = None
        with pdfplumber.open(pdfFile) as pdf:
            lstWordsFromPdf = pdf.pages[0].extract_text().split()
        
        imageOri = Image.open(file).convert("RGB")
        (width, height) = imageOri.size
        # Transform image to force Tesseract recognize all text
        #(1198, 2022),Image.BICUBIC: miss Sailing on about
        #image = image.resize((965, 2022), Image.BICUBIC)  #(965, 2022)
        #image = imageOri #shear(imageOri, image_shear)
        imageCV = convert_from_image_to_cv2(imageOri)
        #imageCV = cv2.resize(imageCV, None, fx=1.15, fy=1.15)
        imageCV = cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY)
        image = convert_from_cv2_to_image(imageCV)
        
        tableContainerSealNo = None
        strContainerSealNo = ""
        strPackingMethod = ""
        totalUnit = ""
        strNW = ""
        strGW = ""
        try:
            if "PACKING LIST" in file:  #based on file 'TCUSLA-2206-02 2(YARN 345)CONTI SAVA 020622 XKD.XLS'
                halfBottomImage = image.crop((0, int(height/2), width, height))
                tableContainerSealNo = tablePred.predict(halfBottomImage)[-1]
                tableContainerSealNo.columns = tableContainerSealNo.iloc[0]
                tableContainerSealNo = tableContainerSealNo[1:]
                if strContainerSealNo == "" and len(tableContainerSealNo.columns) == 5 :
                    if 'net' in tableContainerSealNo.columns[3].lower() and 'gross' in tableContainerSealNo.columns[4].lower():
# =============================================================================
#                     						PACKING DETAILS							
#                     							
#                     ITEM					                ROLL NO.	LENGTH	    NET W'T		GROSS W'T
#                     1000/2,50.0/5*144.0*2401M (2C15GC)	1-16		38,367 M	13,422 KGS	13,790 KGS
#                     1500/2,51.1/5*148.0*2400M (2C52)	    1-7		    16,841 M	9,400 KGS	9,554 KGS
#                     -----------------------------------------------------------------------------------------------------
#                     Total: 					            23		    55,208 M	22,822 KGS	23,344 KGS
# 
# =============================================================================
                        strPackingMethod = tableContainerSealNo.columns[1].split(' ')[0]
                        totalUnit = tableContainerSealNo[tableContainerSealNo.columns[1]].iat[-1]
                        strNW = tableContainerSealNo[tableContainerSealNo.columns[3]].iat[-1]
                        strGW = tableContainerSealNo[tableContainerSealNo.columns[4]].iat[-1]
                    else:                         
                         #     Container	  Seal		    Lbs	       Pallet(s)	 Amount	
                         #    -------------------------------------------------------------
                         #    U0164232  |	VN1624754A  |	40,401  |  34	     |	57,561.97
                         #    -------------------------------------------------------------
                         #    U0499688  |	VN1359581A  |	40,401	|  34	     |	57,561.97	
                         #    ----------------------------------------------------------------
                         #    TOTAL				             80,802	   68		    115,123.93
                        tableContainerSealNo[r'Container']=tableContainerSealNo[r'Container'].apply(lambda x : difflib.get_close_matches(x, lstWordsFromPdf)[0] if str(x) != 'nan' else x)
                        tableContainerSealNo[r'Seal']=tableContainerSealNo[r'Seal'].apply(lambda x : difflib.get_close_matches(x, lstWordsFromPdf)[0] if str(x) != 'nan' else x)
                        tableContainerSealNo[r'Container \ Seal No'] = tableContainerSealNo[r'Container'] + " / " + tableContainerSealNo[r'Seal']
                        strContainerSealNo = '\n'.join(tableContainerSealNo[~tableContainerSealNo[r'Container \ Seal No'].isnull()][r'Container \ Seal No']) 
                        strPackingMethod = tableContainerSealNo.columns[3].split(' ')[0]
                        totalUnit = tableContainerSealNo[strPackingMethod].iat[-1]
                elif len(tableContainerSealNo.columns) == 3 or len(tableContainerSealNo.columns) == 4:
                    if not tableContainerSealNo.filter(regex='ROL').empty:
                        strPackingMethod = 'ROLL'
                        totalUnit = tableContainerSealNo.filter(regex='ROL').dropna().iloc[0][-1].split('-')[-1] #case '1-16' ROLLS
                    elif not tableContainerSealNo.filter(regex='PALL').empty:
                        strPackingMethod = 'PALLET'
                        totalUnit = tableContainerSealNo.filter(regex='PALL').dropna().iloc[0][-1].split('-')[-1]
                    elif not tableContainerSealNo.filter(regex='PAK').empty:
                        strPackingMethod = 'PAKAGE'
                        totalUnit = tableContainerSealNo.filter(regex='PAKAGE').dropna().iloc[0][-1].split('-')[-1]
                        
                    strNW = tableContainerSealNo.filter(regex='NET').dropna().iloc[0][-1]
                    strGW = tableContainerSealNo.filter(regex='GROS').dropna().iloc[0][-1]
                else:
                    #raise Exception("Not the right format Container | Seal | Lbs | Roll | Amount")
                    strContainerSealNo = getContainerSealNo_fromPdf(pdfFile)                                   

                print("strContainerSealNo UNIT PackingMethod:  ",strContainerSealNo + " " + totalUnit + " " + strPackingMethod )
        except Exception as e:  #in case file doesn't have table of Container \ Seal No
            strContainerSealNo = ""
            try: #based on file 'TCDKO-2207-19 200722 (VP5) -02 PET-XKD.XLS'
                if "PACKING LIST" in file: #maybe this file is for PackingMethod: ROLL
                    halfBottomImage = imageOri.crop((0, int(height/2), width, height))
                    tableRolls = tablePred.predict(halfBottomImage)
                    tableRoll = tableRolls[-1]
                    tableRoll.columns = tableRoll.iloc[0]
                    if 'ROL' in tableRoll.columns[0]: #ROL is just for make sure :)
                        strPackingMethod = 'ROLL'
                        totalUnit = tableRoll[tableRoll.columns[0]].iat[-1]
                        if bool(re.match('[\d\w]+',tableRoll[tableRoll.columns[1]].iat[-1])):
                            strNW = tableRoll[tableRoll.columns[1]].iat[-1]
                        else:
                            strNW = tableRoll[tableRoll.columns[1]].iat[-2]
                        if bool(re.match('[\d\w]+',tableRoll[tableRoll.columns[2]].iat[-1])):
                            strGW = tableRoll[tableRoll.columns[2]].iat[-1]
                        else:
                            strGW = tableRoll[tableRoll.columns[2]].iat[-2]
                        print("strPackingMethod totalUnit NetW GrossW:  ",strPackingMethod + " " + totalUnit + " " + strNW + " " + strGW)
                    else:
                        strContainerSealNo = getContainerSealNo_fromPdf(pdfFile) 
            except Exception as e:  #in case file doesn't have table of Container \ Seal No
                print("invoice_info_extract_from_png - PACKING LIST", e)
                strContainerSealNo = getContainerSealNo_fromPdf(pdfFile)   
                pass
            pass
        
        halfLeftImage = image.crop((0, 100, int(width/2), int(height*0.65)))
        halfTopImage = image.crop((0, 0, width, int(height/2)))
        halfBottomImage = image.crop((0, int(height/2)-300, width, height))
        dfFull = do_extract_from_png(image, file+'_full', feature_extractor, tokenizer, model, graph, lstHasToBeQuestion, lstHasToBeAnswer, extQuestionAnswerGraphFunc, lstWordsFromPdf)
        df = dfFull
        notifyConsigneePD = None
        if cut_image:
            for cut_image_item in cut_image:
                if cut_image_item == "left":            
                    dfLeft = do_extract_from_png(halfLeftImage, file+'_left', feature_extractor, tokenizer, model, graph, lstHasToBeQuestion, lstHasToBeAnswer, extQuestionAnswerGraphFunc, lstWordsFromPdf)
                    
                    containerSealNoDf =  dfLeft.loc[dfLeft['question'].str.contains('Container|CONTAINER|Seal|SEAL')]
                    dfLeft =  dfLeft.drop(containerSealNoDf.index)
                    #containerSealNoDf['answer'] = containerSealNoDf.groupby(['question'])['answer'].transform(lambda x :  '\n abc'.join(x) ).drop_duplicates()
                    
                    #Having format as "ContainerNo / SealNo"
                    try:
                        containerSealNoDf['answer'] = containerSealNoDf.loc[containerSealNoDf['answer'].str.contains('/')] \
                                                    .groupby(['question'])['answer'] \
                                                    .apply(lambda x :  '\n'.join(x))[0]
                    except:
                        print('')
                    containerSealNoDf = containerSealNoDf.drop_duplicates()
                    dfLeft = dfLeft.append(containerSealNoDf)
                    if containerSealNoDf.shape[0] > 0:
                        toBeRemovedDF = df.loc[df['question'].str.contains('Container|CONTAINER|Seal|SEAL')]
                        df =  df.drop(toBeRemovedDF.index)
                    
                    df_NotifyConsignee =  df.loc[df['question'].str.contains('Notify|Messrs')].drop_duplicates() 
                    df = df.loc[~df['question'].str.contains('Notify|Messrs')]
                    df_NotifyConsignee['answer'] = df_NotifyConsignee.groupby(['question'])['answer'].transform(lambda x : '\n'.join(x))
                    df_NotifyConsignee = df_NotifyConsignee.drop_duplicates()                    
                    dfLeft_NotifyConsignee =  dfLeft.loc[dfLeft['question'].str.contains('Notify|Messrs')].drop_duplicates() 
                    dfLeft = dfLeft.loc[~dfLeft['question'].str.contains('Notify|Messrs')]
                    dfLeft_NotifyConsignee['answer'] = dfLeft_NotifyConsignee.groupby(['question'])['answer'].transform(lambda x : '\n'.join(x))
                    dfLeft_NotifyConsignee = dfLeft_NotifyConsignee.drop_duplicates()                    
                    notifyConsigneePD = pd.merge(dfLeft_NotifyConsignee, df_NotifyConsignee, how='outer', left_on = ['question'], right_on = ['question']).drop_duplicates()                    
                    notifyConsigneePD['answer'] = ""
                    for index, row in notifyConsigneePD.iterrows():
                        try:
                            notifyConsigneePD.at[index,'answer'] = row['answer_x'] if len(row['answer_x']) > len(row['answer_y']) else row['answer_y']
                        except:
                            notifyConsigneePD.at[index,'answer'] = row['answer_x'] if checkNanValue(row['answer_y']) else row['answer_y']
                            continue
                    notifyConsigneePD = notifyConsigneePD[['question', 'answer']].drop_duplicates()
                    
                    df = pd.merge(dfLeft, df, how='outer', left_on = ['question','answer'], right_on = ['question','answer'])                    
                elif cut_image_item == "top":            
                    dfTop = do_extract_from_png(halfTopImage, file+'_top', feature_extractor, tokenizer, model, graph, lstHasToBeQuestion, lstHasToBeAnswer, extQuestionAnswerGraphFunc)
                    df = pd.merge(dfTop, df, how='outer', left_on = ['question','answer'], right_on = ['question','answer'])
                elif cut_image_item == "bottom":              
                    dfBottom = do_extract_from_png(halfBottomImage, file+'_bottom', feature_extractor, tokenizer, model, graph, lstHasToBeQuestion, lstHasToBeAnswer, extQuestionAnswerGraphFunc, lstWordsFromPdf)
                    desOfGoodsDf =  dfBottom.loc[dfBottom['question'].str.contains('Description|Goods|goods')]
                    #notDesOfGoodsDf =  dfBottom.loc[~dfBottom['question'].isin(['Description of Goods'])]
                    notDesOfGoodsDf =  dfBottom.drop(desOfGoodsDf.index) #Quantity|Net|NET|Gross|GROSS|AMOUNT
                    filter2 = notDesOfGoodsDf['answer'].str.contains('KGS') #only get KGS, not other ~LBS ~Ib
                    filter3 = filter3 = notDesOfGoodsDf['answer'].str.contains('[\(]*[\d]+[ ]*[Ib|LBS|LB][\)]*') == False
                    notDesOfGoodsDf = notDesOfGoodsDf.where(filter3).dropna()                
                    containerSealNoDf =  notDesOfGoodsDf.loc[notDesOfGoodsDf['question'].str.contains('Container|CONTAINER|Seal|SEAL')]
                    notDesOfGoodsDf =  notDesOfGoodsDf.drop(containerSealNoDf.index)
                    #Having format as "ContainerNo / SealNo"
                    try:
                        containerSealNoDf['answer'] = containerSealNoDf.loc[containerSealNoDf['answer'].str.contains('/')] \
                                                    .groupby(['question'])['answer'] \
                                                    .apply(lambda x :  '\n'.join(x))[0]
                    except:
                        print('')
                    containerSealNoDf = containerSealNoDf.drop_duplicates()
                    notDesOfGoodsDf = notDesOfGoodsDf.append(containerSealNoDf)
                    
                    #g =  notDesOfGoodsDf.groupby(['question'])
                    #notDesOfGoodsDf = (pd.concat([g.tail(1)]).drop_duplicates().sort_values('question').reset_index(drop=True)) #get the last row of each group
                                        
                    #desOfGoodsDf = desOfGoodsDf if desOfGoodsDf.shape[0] % 2 == 0 else desOfGoodsDf[1:]
                    desOfGoodsDf = desOfGoodsDf[~desOfGoodsDf["answer"].str.contains("PORT|ORIGIN|PO NO|NCM NO|(PLACE OF TERMS OF PRICE)")]
                    desOfGoodsDf = desOfGoodsDf[desOfGoodsDf["answer"].str.len() >= 15]
                    desOfGoodsDf['answer'] = desOfGoodsDf.groupby(['question'])['answer'].transform(lambda x : '\n'.join(x))                
                    desOfGoodsDf = desOfGoodsDf.drop_duplicates()
                    dfBottom = notDesOfGoodsDf.append(desOfGoodsDf)
                    dfBottomColumns = dfBottom['question']
                    
                    l = ['Description','Quantity','Net','NET','Gross','GROSS','AMOUNT']
                    l = ['Description', 'AMOUNT']
                    tobeRemoveQuestion = []
                    for w in l:
                        if any(w in q for q in dfBottom['question']):
                            tobeRemoveQuestion.append(w)
                    tobeRemoveQuestion = '|'.join(map(str,tobeRemoveQuestion))
                    if len(tobeRemoveQuestion) > 1:
                        tobeRemoved = df.loc[df['question'].str.contains(tobeRemoveQuestion)]
                        df = df.drop(tobeRemoved.index)
                    #In some cases, dfBottom doesn't have DescriptionOfGood
                    try:
                        if desOfGoodsDf.shape[0] == 0:
                            desOfGoodsDf_x =  df.loc[df['question'].str.contains('Description|Goods|goods')][['question','answer_x']].drop_duplicates()
                            desOfGoodsDf_x['answer_x'] =  desOfGoodsDf_x.groupby(['question'])['answer_x'].transform(lambda x : '\n'.join(x)).drop_duplicates()
                            desOfGoodsDf_x.dropna(inplace =True)
                            desOfGoodsDf_x.rename({'answer_x':'answer'}, axis=1,inplace=True)
                            desOfGoodsDf_x.reset_index(drop=True, inplace=True)
                            
                            desOfGoodsDf_y =  df.loc[df['question'].str.contains('Description|Goods|goods')][['question','answer_y']].drop_duplicates()
                            desOfGoodsDf_y['answer_y'] =  desOfGoodsDf_y.groupby(['question'])['answer_y'].transform(lambda x : '\n'.join(x)).drop_duplicates()
                            desOfGoodsDf_y.dropna(inplace =True)
                            desOfGoodsDf_y.rename({'answer_y':'answer'}, axis=1,inplace=True)
                            desOfGoodsDf_y.reset_index(drop=True, inplace=True)
                            
                            desOfGoodsDf = desOfGoodsDf_x if len(desOfGoodsDf_x['answer'][0]) >= len(desOfGoodsDf_y['answer'][0]) else desOfGoodsDf_y
                            df = df.loc[~df['question'].str.contains('Description|Goods|goods')]
                    except:
                        pass
                        
                    #df = pd.merge(notDesOfGoodsDf.append(desOfGoodsDf), df, how='outer', left_on = 'question', right_on = 'question')                    
                    df = pd.merge(notDesOfGoodsDf.append(desOfGoodsDf), df, how='outer', left_on = ['question','answer'], right_on = ['question','answer'])

        try:
            df['answer_x'] = df['answer_x'] .fillna(df['answer_y'] )
            df['answer'] = df['answer'] .fillna(df['answer_x'] )            
            df = df[['question', 'answer']].drop_duplicates()
        except:
            df['answer'] = df.filter(like='answer').ffill(axis=1).iloc[:,-1]
        if notifyConsigneePD is not None:
            df = df.append(notifyConsigneePD[['question', 'answer']])
        df['question2header'] = ""
        df['sort'] = ""
        if strContainerSealNo:
            tobeRemovedDf =  df.loc[df['question'].str.contains('Container|Seal|UNIT|PackingMethod')]
            df =  df.drop(tobeRemovedDf.index)
            try:
                totalUnit = float(totalUnit)
            except:
                pass
            t = pd.DataFrame({'question':["Container \ Seal No", "UNIT", "PackingMethod"],
                              'question2header':["Container \ Seal No", "UNIT", "PackingMethod"], 
                              'answer':[strContainerSealNo, totalUnit, strPackingMethod], 
                              'sort':[reverseSwitcher.get("Container \ Seal No", 888), reverseSwitcher.get("UNIT", 888), reverseSwitcher.get("PackingMethod", 888)]})
            df = df.append(t)
        elif 'ROLL' in strPackingMethod.upper() or 'PALLET' in strPackingMethod.upper() or 'PACKAGE' in strPackingMethod.upper():
            try:
                totalUnit = float(totalUnit)
            except:
                pass
            t = pd.DataFrame({'question':["PackingMethod", "UNIT", "N.W", "G.W"],
                              'question2header':["PackingMethod", "UNIT", "N.W", "G.W"], 
                              'answer':[strPackingMethod, totalUnit, strNW, strGW], 
                              'sort':[reverseSwitcher.get("PackingMethod", 888), reverseSwitcher.get("UNIT", 888), reverseSwitcher.get("N.W", 888), reverseSwitcher.get("G.W", 888)]})
            df = df.append(t)
            
        df.reset_index(drop=True, inplace=True)        
        dfFinal = question2header_func(df)
       
                
        out_collection[os.path.basename(file)] = dfFinal[['question', 'question2header', 'answer', 'sort']]
        



