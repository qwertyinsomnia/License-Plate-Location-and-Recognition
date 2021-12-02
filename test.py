import os


for root, dirs, files in os.walk("..\\..\\License-Plate-Recognition-master\\train\\chars2"):
   if len(os.path.basename(root)) > 1:
      continue
   root_int = ord(os.path.basename(root))
   print(files)
   for filename in files:
      filepath = os.path.join(root,filename)
      if filename.startswith("gt_"):
         print(filepath)
      # digit_img = cv2.imread(filepath)
      # digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
      # chars_train.append(digit_img)
      # chars_label.append(root_int)