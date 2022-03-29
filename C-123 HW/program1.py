import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plot
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from types import new_class

X = np.load('image-C-122 HW.npz')['arr_0']
y = pd.read_csv('data - C-122 HW.csv')["labels"]

print(pd.Series(y).value_counts())

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

nclasses = len(classes)


from pickle import FALSE
samplesPerClass = 5

figure = plot.figure(figsize = (nclasses*2, (1 + samplesPerClass*2)))

idx_cls = 0

for cls in classes :
  idxs = np.flatnonzero(y == cls)
  idxs = np.random.choice(idxs, samplesPerClass, replace = False)

  i = 0

  for idx in idxs :
    plot_idx = i*nclasses + idx_cls + 1

    p = plot.subplot(samplesPerClass, nclasses, plot_idx)
    p = sns.heatmap(np.array(X[idx]).reshape(22, 30), cmap = plot.cm.gray, xticklabels = False, yticklabels = False, cbar = False)
    p = plot.axis("off")

    i+=1

  idx_cls+=1

xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

xtrainScaled = xtrain/255.0

xtestScaled = xtest/255.0

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(xtrainScaled, ytrain)

ypredict = clf.predict(xtestScaled)

accuracy = accuracy_score(ytest, ypredict)

print(accuracy)

cap = cv2.VideoCapture(0)

while(True):
  # Capture frame-by-frame
  try:
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Drawing a box in the center of the video
    height, width = gray.shape
    upper_left = (int(width / 2 - 56), int(height / 2 - 56))
    bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

    #To only consider the area inside the box for detecting the digit
    #roi = Region Of Interest
    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    #Converting cv2 image to pil format
    im_pil = Image.fromarray(roi)

    # convert to grayscale image - 'L' format means each pixel is 
    # represented by a single value from 0 to 255
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    print("Predicted class is: ", test_pred)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  except Exception as e:
    pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()