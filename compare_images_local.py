from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json

file_name1 = "../Images/design/Facebook_loginorsignup_design.png"
file_name2 = "../Images/actual/Facebook_loginorsignup_actual.png"

im1 = np.array(Image.open(file_name1), dtype=np.uint8)
im2= np.array(Image.open(file_name2), dtype=np.uint8)


w=10
h=10
fig=plt.figure(figsize=(1, 1))
img = np.random.randint(10, size=(h,w))
ax1=fig.add_subplot(1, 2, 1)
plt.imshow(im1)
ax2=fig.add_subplot(1, 2, 2)
plt.imshow(im2)

    
# Now there is a trained endpoint that can be used to make a prediction
prediction_key="446e20485dfd4d6f9efa962724bb71cb"
iteration = "Iteration5"
#ENDPOINT="https://southcentralus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/c6d6869a-a1e5-45a5-a3d6-dc6d7805e2fd/detect/iterations/Iteration1/image"
#https://southcentralus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/671a4689-f3e3-4b78-a634-9ae80aaf635e/detect/iterations/Iteration1/image 
ENDPOINT = "https://southcentralus.api.cognitive.microsoft.com"

predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)

with open(file_name1, "rb") as image_contents:
    results1 = predictor.classify_image("671a4689-f3e3-4b78-a634-9ae80aaf635e", iteration, image_contents.read())

# Display the results.    
#for prediction in results.predictions:
    #print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))



with Image.open(file_name1) as image: 
	width1, height1 = image.size 


for prediction in results1.predictions:
    if prediction.probability * 100 > 30:
        rect = patches.Rectangle((prediction.bounding_box.left * width1,prediction.bounding_box.top * height1),prediction.bounding_box.width * width1,prediction.bounding_box.height * height1,linewidth=1,edgecolor='g',facecolor='none')
        #plt.text(prediction.bounding_box.left * width1,prediction.bounding_box.top * height1-25, prediction.tag_name,fontsize=6)
        ax1.add_patch(rect)
        

#----------------------------------------------------------
with open(file_name2, "rb") as image_contents:
    results2 = predictor.classify_image("671a4689-f3e3-4b78-a634-9ae80aaf635e", iteration, image_contents.read())



with Image.open(file_name2) as image: 
	width2, height2 = image.size 


#for prediction in results2.predictions:
    #if prediction.probability * 100 > 30:
        #rect = patches.Rectangle((prediction.bounding_box.left * width2,prediction.bounding_box.top * height2),prediction.bounding_box.width * width2,prediction.bounding_box.height * height2,linewidth=1,edgecolor='blue',facecolor='none')
        #ax2.add_patch(rect)


#-----------------------Comparing Intersects

for prediction1 in results1.predictions:
    match_found=0
    if prediction1.probability * 100 > 30:
        #rect1 = patches.Rectangle(((prediction1.bounding_box.left * width1)+40,(prediction1.bounding_box.top * height1)-40),(prediction1.bounding_box.width * width1)+40,(prediction1.bounding_box.height * height1)-40,linewidth=1,edgecolor='red',facecolor='none')
        #ax2.add_patch(rect1)
        
        for prediction2 in results2.predictions:
            if prediction2.probability * 100 > 30:
                
                #print("\t" + prediction1.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction1.probability * 100, prediction1.bounding_box.left*width1-40, prediction1.bounding_box.top*height1-40, prediction1.bounding_box.width*width1+80, prediction1.bounding_box.height*height1+80))
                #print("\t" + prediction2.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction2.probability * 100, prediction2.bounding_box.left*width2, prediction2.bounding_box.top*height2, prediction2.bounding_box.width*width2, prediction2.bounding_box.height*height2))
                #rect = patches.Rectangle((prediction1.bounding_box.left*width1-40, prediction1.bounding_box.top*height1-40), prediction1.bounding_box.width*width1+80, prediction1.bounding_box.height*height1+80,linewidth=1,edgecolor='red',facecolor='none')
                #ax2.add_patch(rect)

                if ((prediction1.bounding_box.left * width1) == prediction2.bounding_box.left * width2 \
                    and (prediction1.bounding_box.top * height1) == prediction2.bounding_box.top * height2 and \
                    (prediction1.bounding_box.width * width1 ) == prediction2.bounding_box.width * width2 and \
                    (prediction1.bounding_box.height * height1 )  == prediction2.bounding_box.height * height2):
                    if (prediction1.tag_name == prediction2.tag_name):
                        #print("I am here")
                        match_found = 1
                        break
                elif ((prediction1.bounding_box.left * width1 - 40) < prediction2.bounding_box.left * width2 \
                    and (prediction1.bounding_box.top * height1 - 40) < prediction2.bounding_box.top * height2 and \
                    (prediction1.bounding_box.width * width1 + 80) > prediction2.bounding_box.width * width2 and \
                    (prediction1.bounding_box.height * height1 + 80)  > prediction2.bounding_box.height * height2):
                    if (prediction1.tag_name == prediction2.tag_name):
                        #print("--------HERE---------------")
                        #rect = patches.Rectangle((prediction2.bounding_box.left * width2,prediction2.bounding_box.top * height2),prediction2.bounding_box.width * width2,prediction2.bounding_box.height * height2,linewidth=1,edgecolor='green',facecolor='none')
                        #ax2.add_patch(rect)
                        #rect = patches.Rectangle((prediction1.bounding_box.left * width2 - 40,prediction1.bounding_box.top * height2-40),prediction2.bounding_box.width * width2+80,prediction2.bounding_box.height * height2+80,linewidth=1,edgecolor='red',facecolor='none')
                        #ax2.add_patch(rect)
                            
                        match_found=1
                            
                        break
                    #else:
                        #match_found = 0
                        #break



        #if match_found==1:
            #rect = patches.Rectangle((prediction1.bounding_box.left * width1,prediction1.bounding_box.top * height1),prediction1.bounding_box.width * width1,prediction1.bounding_box.height * height1,linewidth=1,edgecolor='green',facecolor='none')
            #plt.text(prediction1.bounding_box.left * width1,prediction1.bounding_box.top * height1-5, prediction1.tag_name + " - Match Found",fontsize=6)
            #ax2.add_patch(rect)
            
            #rect = patches.Rectangle((prediction1.bounding_box.left * width1,prediction1.bounding_box.top * height1),prediction1.bounding_box.width * width1,prediction1.bounding_box.height * height1,linewidth=1,edgecolor='green',facecolor='none')
            #ax1.add_patch(rect)    

        
        if match_found==0:
            rect = patches.Rectangle((prediction1.bounding_box.left * width1,prediction1.bounding_box.top * height1),prediction1.bounding_box.width * width1,prediction1.bounding_box.height * height1,linewidth=1,edgecolor='red',facecolor='none')
            plt.text(prediction1.bounding_box.left * width1,prediction1.bounding_box.top * height1-5, prediction1.tag_name + "[" + str(round(prediction1.probability*100,2)) + "%] missing",fontsize=6)
            ax2.add_patch(rect)
            
            #rect = patches.Rectangle((prediction1.bounding_box.left * width1,prediction1.bounding_box.top * height1),prediction1.bounding_box.width * width1,prediction1.bounding_box.height * height1,linewidth=1,edgecolor='red',facecolor='none')
            #ax1.add_patch(rect)    
        

#mng = plt.get_current_fig_manager()
#mng.full_screen_toggle()

figure = plt.gcf() # get current figure
figure.set_size_inches(8, 6)
# when saving, specify the DPI

plt.savefig(r"../Images/results/Facebook_loginorsignup_result.jpg",bbox_inches='tight',quality=100, dpi=800)
plt.show()

