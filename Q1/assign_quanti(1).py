# def euclid(frequent,pixel):
#   distances = []
#   for i in frequent:
#     point_a = np. array(pixel)
#     point_b = np. array(i[:3])
#     distance = np. linalg. norm(point_a - point_b)
#     distances.append(distance)
#   mini = min(distances)
#   index = distances.index(mini)
#   pix = frequent[index]
#   return pix[:3]



checker = []
for i in frequent:
  f1 = ''
  for j in range(len(i)-1):
      if(j==0):
        f1+=str(i[j])
      else:
        f1+= ':'+str(i[j])
  checker.append(f1)
print(checker)

def popular_img(img):
  new_arr=np.zeros(img.shape,dtype=np.int8)
  #print(new)
  img_arr = np.asarray(img)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      pixel=img_arr[i][j]
      #print(pixel)
      s1=str(pixel[0])+':'+str(pixel[1])+':'+str(pixel[2])
      #print(s1)
      if s1 in checker:
        new_arr[i][j] = pixel
      else:
        new_arr[i][j] = euclid(frequent,pixel)
  return new_arr

new_arr = popular_img(img)
plt.imshow(new_arr)

"""# Median Cut"""

img = np.array(Image.open('/content/kasturi-roy-x33dnDTe2QQ-unsplash.jpg'))
plt.imshow(img)

def colors_and_counts(img):
  color_map = {}
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      color = str(img[i][j][0]) + str(img[i][j][1]) + str(img[i][j][2])
      # print(str(color))
      if str(color) in color_map.keys():
        # color_map[str(color)]['i'].append([img[i][j][0], img[i][j][1], img[i][j][2]])
        color_map[str(color)][3] += 1
        color_map[str(color)][-1].append((i, j))
      else:
        color_map.update({str(color):[img[i][j][0], img[i][j][1], img[i][j][2], 1, [(i, j)]]})
  print('Getting colors and counts.....')
  return list(color_map.values())

cnc = colors_and_counts(img)

cnc

def find_max_box_dimension(b):
  sizer = b[3] - b[2]
  sizeg = b[5] - b[4]
  sizeb = b[7] - b[6]
  print('Finding maximum box dimension')
  if (sizer >= sizeg) and (sizer >= sizeb):
   return 0
  elif (sizeg >= sizer) and (sizeg >= sizeb):
    return 1
  else:
    return 2

def create_color_box(C, level):
  # colors = np.array(C)[:,:-1]
  colors = np.array(C)
  rmin = np.min(colors[:,0])
  rmax = np.max(colors[:,0])
  gmin = np.min(colors[:,1])
  gmax = np.max(colors[:,1])
  bmin = np.min(colors[:,2])
  bmax = np.max(colors[:,2])
  print('Creating color box with level: ', level)
  return [colors.tolist(), level, rmin, rmax, gmin, gmax, bmin, bmax]

def find_box_to_split(B):
  bs = []
  print('Finding box to split....')
  for b in B:
    if len(b[0]) >= 2:
      bs.append(b)
  if len(bs) == 0:
    return
  else:
    minl = float('inf')
    box = bs[0]
    for b in B:
      if b[1] < minl:
        minl = b[1]
        box = b
    return box

def split_box(b):
  m = b[1]
  d = find_max_box_dimension(b)
  C = b[0]
  C = np.array(C)
  C = C[C[:, d].argsort()]
  # C = C.tolist()
  median_index = int((len(C) + 1) / 2)
  c1 = C[:median_index]
  c2 = C[median_index:]

  # print(c1)
  # print(c2)

  print('Splitting the box.....')
  return (create_color_box(c1, m+1), create_color_box(c2, m+1))

def average_colors(b):
  C = b[0]
  n = 0
  rsum, gsum, bsum = 0, 0, 0
  for c in C:
    k = c[-2]
    n+=k
    rsum+=(k*c[0])
    gsum+=(k*c[1])
    bsum+=(k*c[2])
  ravg, gavg, bavg = rsum//n, gsum//n, bsum//n
  print('Finding the average of the colors.....')
  return [ravg, gavg, bavg]

def find_representative_colors(image, km, C):
  image2 = image.copy()
  if len(C) <= km:
    return C
  else:
    level = 0
    b0 = create_color_box(C, level)
    B = []
    B.append(b0)
    k = 1
    done = False
    while (k < km) and not done:
      b = find_box_to_split(B)
      if len(b)!=0:
        b1, b2 = split_box(b)
        B.append(b1)
        B.append(b2)
        B.remove(b)
        k+=1
      else:
        done = True
    cr = []
    for b in B:
      avg = average_colors(b)
      cr.append(avg)
      C = b[0]
      # print(avg)
      # print(C)
      for i in range(len(C)):
        for ri, ci in C[i][-1]:
          image2[ri][ci] = avg
      
    print('Finding CR values....')
    return image2, cr

# def quant_img(image, cr):
#   image2 = image.copy()
#   min = float('inf')
#   for i in range(image2.shape[0]):
#     for j in range(image2.shape[1]):
#       for c in cr:
#         dist = np.sqrt(np.sum(np.square(c-image[i][j])))
#         if dist < min:
#           min = dist
#           point = c
#       image2[i][j] = point
#   print('Quantizing the image.....')
#   return image2

def median_cut(image, km):
  C = colors_and_counts(image)
  # print(C)
  median, cr = find_representative_colors(image, km, C)
  return median, cr

median,  cr = median_cut(img, 4)
plt.imshow(median)

median,  cr = median_cut(img, 16)
plt.imshow(median)

cr

median_cnc = colors_and_counts(median)

median_colors = []
for j in median_cnc:
  median_colors.append(j[:3])
median_colors

imsave('/content/median4.png', median)
imsave('/content/img.png', img)

"""# Floyd Steinberg dithering """

img = cv2.imread('/content/offset_comp_772626-opt.jpg')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
from PIL import Image
#img = np.array(Image.open('/content/kasturi-roy-x33dnDTe2QQ-unsplash.jpg'))
img = np.array(Image.open('/content/images.jpg'))
plt.imshow(img)

#def popular_img(img):
new_arr=np.zeros(img.shape,dtype=np.float32)
#print(new)
img_arr = np.asarray(img)
for i in range(img.shape[0]-1):
  for j in range(img.shape[1]-1):
    pixel=img_arr[i][j]
    #print(pixel)
    s1=str(pixel[0])+':'+str(pixel[1])+':'+str(pixel[2])
    #print(s1)
    if s1 in checker:
      new_arr[i][j] = pixel
    else:
      new_arr[i][j] = euclid(frequent,pixel)
      #quant error
      point_a = np. array(pixel)
      point_b = np. array(new_arr[i][j])
      diff = (point_a - point_b)**2
      error = diff[0]+diff[1]+diff[2]
      #print(error)
      (new_arr[i][j+1]) = np.array(img_arr[i][j+1]) + (error*3/8)
      (new_arr[i+1][j]) = np.array(img_arr[i+1][j]) + (error*3/8)
      (new_arr[i+1][j+1]) = np.array(img_arr[i+1][j+1]) + (error/4)
      #return new_arr

final = new_arr.astype(np.uint8)
plt.imshow(final)









