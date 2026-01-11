# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# 2024/06/26
# ImageMaskDatasetGenerator.py

import os
import shutil
import cv2
import numpy as np

import glob
import xmltodict
import json
import math
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import traceback

class ImageMaskDatasetGenerator:

  def __init__(self, images_dir, xmls_dir, jsons_dir, output_dir, 
                rename = True,
                resize = False,
                augmentation=True, debug=False):
     self.seed       = 137
     self.rename     = rename
     self.resize     = resize
     SIZE            = 512
     self.W          = SIZE
     self.H          = SIZE
     self.size       = (SIZE, SIZE)
     self.debug      = debug

     self.images_dir = images_dir
     self.xmls_dir   = xmls_dir 
     self.jsons_dir  = jsons_dir
     self.output_dir = output_dir
     
     self.output_images_dir = output_dir + "/images"
     self.output_masks_dir  = output_dir + "/masks"
     os.makedirs(self.output_images_dir)
     os.makedirs(self.output_masks_dir)
     
     if os.path.exists(self.jsons_dir):
      shutil.rmtree(self.jsons_dir)
     os.makedirs(self.jsons_dir)

     self.augmentation = augmentation
     if self.augmentation:
       self.hflip    = True
       self.vflip    = True
       self.rotation = True
       self.ANGLES   = [20, 400, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320]

       self.deformation=True
       self.alpha    = 1300
       self.sigmoids = [8.0, 10.0]
          
       self.distortion=True
       self.gaussina_filer_rsigma = 40
       self.gaussina_filer_sigma  = 0.5
       self.distortions           = [0.02, 0.03,]
       self.rsigma = "sigma"  + str(self.gaussina_filer_rsigma)
       self.sigma  = "rsigma" + str(self.gaussina_filer_sigma)
      
       self.shrinking = True
       self.SHRINKS   = [0.7, 0.8, 0.9]

       self.barrel_distortion = True
       self.radius     = 0.3
       self.amount     =  0.3
       self.centers    = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]

       self.pincussion_distortion = True
       self.pinc_radius  = 0.3
       self.pinc_amount  = -0.3
       self.pinc_centers = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]

  def generate(self):
     self.convert_tif_to_jpg()
     self.convert_xml_to_mask()
     
  def convert_tif_to_jpg(self) : 
     tif_files = glob.glob(self.images_dir + "/*.tif")
     tif_files = sorted(tif_files)
     index = 10000
     for tif_file in tif_files:
       index += 1
       image = cv2.imread(tif_file)
       #self.split_tiles(image, index, self.output_images_dir)

       basename = os.path.basename(tif_file)
       if self.rename:
         #basename = str(index) + ".jpg"
         jpgfile  = str(index) + ".jpg"
       else:
         jpgfile  = basename.replace(".tif", ".jpg")
       output_filepath = os.path.join(self.output_images_dir, jpgfile)
       if self.resize:
         image = cv2.resize(image, self.size)
       
       cv2.imwrite(output_filepath, image)
       print("--- Saved {}".format(output_filepath))
       if self.augmentation:
         self.augment(image, jpgfile, self.output_images_dir, border=(0, 0, 0), mask=False)
         
  def convert_xml_to_mask(self):
    xml_files = glob.glob(self.xmls_dir + "/*.xml")
    xml_files = sorted(xml_files)
    index = 10000
    for xml_file in xml_files:
      index += 1
      print("=== Convert xml_file {}".format(xml_file))
      basename = os.path.basename(xml_file)
      if self.rename:
        jpg_file = str(index) + ".jpg"
      else:
        jpg_file = basename.replace(".xml", ".jpg")

      with open(xml_file, 'r') as file:
        xml_data = file.read()
        json_data   = xmltodict.parse(xml_data)
        json_string = json.dumps(json_data, indent=4)
        json_string = json_string.replace("@", "")
        data        = json.loads(json_string)
        self.parse_json(data, jpg_file, self.output_masks_dir)
        if self.debug:
          json_file = basename.replace(".xml", ".json")
          json_filepath = os.path.join(self.jsons_dir, json_file)
          with open(json_filepath, 'w') as f:
            f.writelines(json_string)
            print("=== Saved {}".format(json_filepath))

  def find_regions(self, data):
    regions = None
    try:
      Annotations = data.get("Annotations")
      Annotation  = Annotations.get("Annotation")
      if type(Annotation) == list:
        print("     Annotation is list")
        Annotation = Annotation[0]
      Regions     = Annotation.get("Regions")
      regions     = Regions.get("Region")
    except:
      traceback.print_exc()
    return regions
  
  def parse_json(self, data, basename, output_dir ):
      regions = self.find_regions(data)
      #regions = data["Annotations"]["Annotation"]["Regions"]["Region"]
      if regions == None or type(regions) !=list:
        return
      
      mask    = np.zeros((1000, 1000, 3))
      for region in regions:
        vertex = region["Vertices"]["Vertex"]
        points = []
        for xy in vertex:
          try:
            x = float(xy["X"])
            y = float(xy["Y"])
            points += [[x, y]]
          except:
            traceback.print_exc()
            pass
        if len(points) != 0: 
          # Please see https://stackoverflow.com/questions/17241830/opencv-polylines-function-in-python-throws-exception
          array = np.array(points).reshape((-1,1,2)).astype(np.int32)
          cv2.fillPoly(mask, pts=[array], color=(255, 255, 255))
   
      mask = mask.reshape((1000, 1000, 3))
      if self.resize:
        mask = cv2.resize(mask, self.size)
      output_filepath = os.path.join(output_dir, basename)
      cv2.imwrite(output_filepath, mask)
      print("=== Saved {}".format(output_filepath))
      if self.augmentation:
        self.augment(mask, basename, output_dir, border=(0, 0, 0), mask=True)

  def augment(self, image, basename, output_dir, border=(0, 0, 0), mask=False):
    border = image[2][2].tolist()
    if mask:
      border = (0, 0, 0)
    print("---- border {}".format(border))
    if self.hflip:
      flipped = self.horizontal_flip(image)
      output_filepath = os.path.join(output_dir, "hflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.vflip:
      flipped = self.vertical_flip(image)
      output_filepath = os.path.join(output_dir, "vflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.rotation:
      self.rotate(image, basename, output_dir, border)

    if self.deformation:
      self.deform(image, basename, output_dir)

    if self.distortion:
      self.distort(image, basename, output_dir)

    if self.shrinking:
      self.shrink(image, basename, output_dir, mask)

    if self.barrel_distortion:
      self.barrel_distort(image, basename, output_dir)

    if self.pincussion_distortion:
      self.pincussion_distort(image, basename, output_dir)

  def horizontal_flip(self, image): 
    print("shape image {}".format(image.shape))
    if len(image.shape)==3:
      return  image[:, ::-1, :]
    else:
      return  image[:, ::-1, ]

  def vertical_flip(self, image):
    if len(image.shape) == 3:
      return image[::-1, :, :]
    else:
      return image[::-1, :, ]

  def rotate(self, image, basename, output_dir, border):
    for angle in self.ANGLES:      
      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H), borderValue=border)
      output_filepath = os.path.join(output_dir, "rotated_" + str(angle) + "_" + basename)
      cv2.imwrite(output_filepath, rotated_image)
      print("--- Saved {}".format(output_filepath))

  def deform(self, image, basename, output_dir): 
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    random_state = np.random.RandomState(self.seed)

    shape = image.shape
    for sigmoid in self.sigmoids:
      dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      #dz = np.zeros_like(dx)
      print("------ shape {}".format(shape))
      x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
      indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

      deformed_image = map_coordinates(image, indices, order=1, mode='nearest')  
      deformed_image = deformed_image.reshape(image.shape)

      image_filename = "deformed" + "_alpha_" + str(self.alpha) + "_sigmoid_" +str(sigmoid) + "_" + basename
      image_filepath  = os.path.join(output_dir, image_filename)
      cv2.imwrite(image_filepath, deformed_image)
      print("=== Saved deformed {}".format(image_filepath))

  def distort(self, image, basename, output_dir):
    shape = (image.shape[1], image.shape[0])
    (w, h) = shape
    xsize = w
    if h>w:
      xsize = h
    # Resize original img to a square image
    resized = cv2.resize(image, (xsize, xsize))
 
    shape   = (xsize, xsize)
 
    t = np.random.normal(size = shape)
    for size in self.distortions:
      filename = "distorted_" + str(size) + "_" + self.sigma + "_" + self.rsigma + "_" + basename
      output_file = os.path.join(output_dir, filename)    
      dx = gaussian_filter(t, self.gaussina_filer_rsigma, order =(0,1))
      dy = gaussian_filter(t, self.gaussina_filer_rsigma, order =(1,0))
      sizex = int(xsize*size)
      sizey = int(xsize*size)
      dx *= sizex/dx.max()
      dy *= sizey/dy.max()

      image = gaussian_filter(image, self.gaussina_filer_sigma)

      yy, xx = np.indices(shape)
      xmap = (xx-dx).astype(np.float32)
      ymap = (yy-dy).astype(np.float32)

      distorted = cv2.remap(resized, xmap, ymap, cv2.INTER_LINEAR)
      distorted = cv2.resize(distorted, (w, h))
      cv2.imwrite(output_file, distorted)
      print("=== Saved distorted image file{}".format(output_file))

  def shrink(self, image, basename, output_dir, mask=True):
    h, w = image.shape[:2]
  
    for shrink in self.SHRINKS:
      rw = int (w * shrink)
      rh = int (h * shrink)
      resized_image = cv2.resize(image, dsize= (rw, rh),  interpolation=cv2.INTER_NEAREST)
      
      squared_image = self.paste(resized_image, mask=False)
    
      ratio   = str(shrink).replace(".", "_")
      image_filename = "shrinked_" + ratio + "_" + basename
      image_filepath  = os.path.join(output_dir, image_filename)
      cv2.imwrite(image_filepath, squared_image)
      print("=== Saved shrinked {}".format(image_filepath))

  def paste(self, image, mask=False):
    l = len(image.shape)
   
    h, w,  = image.shape[:2]
    if l==3:
      back_color = image[2][2]
      background = np.ones((self.H, self.W, 3), dtype=np.uint8)
      background = background * back_color

      #background = np.zeros((self.H, self.W, 3), dtype=np.uint8)
      #(b, g, r) = image[h-10][w-10] 
      #print("r:{} g:{} b:c{}".format(b,g,r))
      #background += [b, g, r][::-1]
    else:
      v =  image[h-10][w-10] 
      image  = np.expand_dims(image, axis=-1) 
      background = np.zeros((self.H, self.W, 1), dtype=np.uint8)
      background[background !=v] = v
    x = (self.W - w)//2
    y = (self.H - h)//2
    background[y:y+h, x:x+w] = image
    return background
  
  def barrel_distort(self, image, basename, output_dir):
    distorted_image  = image
    (h, w, _) = image.shape

    # set up the x and y maps as float32
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    scale_x = 1
    scale_y = 1
    index = 100
    for center in self.centers:
      index += 1
      (ox, oy) = center
      center_x = w * ox
      center_y = h * oy
      radius = w * self.radius
      amount = self.amount   
      # negative values produce pincushion
 
      # create map with the barrel pincushion distortion formula
      for y in range(h):
        delta_y = scale_y * (y - center_y)
        for x in range(w):
          # determine if pixel is within an ellipse
          delta_x = scale_x * (x - center_x)
          distance = delta_x * delta_x + delta_y * delta_y
          if distance >= (radius * radius):
            map_x[y, x] = x
            map_y[y, x] = y
          else:
            factor = 1.0
            if distance > 0.0:
                factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
            map_x[y, x] = factor * delta_x / scale_x + center_x
            map_y[y, x] = factor * delta_y / scale_y + center_y
            

       # do the remap
      distorted_image = cv2.remap(distorted_image, map_x, map_y, cv2.INTER_LINEAR)
      if distorted_image.ndim == 2:
        distorted_image  = np.expand_dims(distorted_image, axis=-1) 

      image_filename = "barrdistorted_" + str(index) + "_" + str(self.radius) + "_"  + str(self.amount) + "_" + basename

      image_filepath  = os.path.join(output_dir, image_filename)
      cv2.imwrite(image_filepath, distorted_image)
      print("=== Saved {}".format(image_filepath))

  def pincussion_distort(self, image, basename, output_dir):
    distorted_image  = image
    (h, w, _) = image.shape

    # set up the x and y maps as float32
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    scale_x = 1
    scale_y = 1
    index = 100
    for center in self.pinc_centers:
      index += 1
      (ox, oy) = center
      center_x = w * ox
      center_y = h * oy
      radius = w * self.pinc_radius
      amount = self.pinc_amount   
      # negative values produce pincushion
 
      # create map with the barrel pincushion distortion formula
      for y in range(h):
        delta_y = scale_y * (y - center_y)
        for x in range(w):
          # determine if pixel is within an ellipse
          delta_x = scale_x * (x - center_x)
          distance = delta_x * delta_x + delta_y * delta_y
          if distance >= (radius * radius):
            map_x[y, x] = x
            map_y[y, x] = y
          else:
            factor = 1.0
            if distance > 0.0:
                factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
            map_x[y, x] = factor * delta_x / scale_x + center_x
            map_y[y, x] = factor * delta_y / scale_y + center_y
            

       # do the remap
      distorted_image = cv2.remap(distorted_image, map_x, map_y, cv2.INTER_LINEAR)
      if distorted_image.ndim == 2:
        distorted_image  = np.expand_dims(distorted_image, axis=-1) 

      image_filename = "pincdistorted_" + str(index) + "_" + str(self.pinc_radius) + "_"  + str(self.pinc_amount) + "_" + basename

      image_filepath  = os.path.join(output_dir, image_filename)
      cv2.imwrite(image_filepath, distorted_image)
      print("=== Saved {}".format(image_filepath))


if __name__ == "__main__":
  try:
    output_dir = "./MoNuSeg-master"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    # Train Dataset
    
    train_images_dir = "./MoNuSeg 2018 Training Data/Tissue Images"
    train_xmls_dir   = "./MoNuSeg 2018 Training Data/Annotations"
    train_jsons_dir  = "./train/jsons"
    
    generator = ImageMaskDatasetGenerator(train_images_dir, train_xmls_dir, 
                                          train_jsons_dir, output_dir,
                                          rename = True,
                                          resize = True,
                                          augmentation=True, debug=False)

    generator.generate()
    
 
    # Create mini_test dataset from MoNusegTestData    
    output_dir  = "./mini_test"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    # Test Dataset
    test_images_dir  = "./MoNuSegTestData"
    test_xmls_dir    = "./MoNuSegTestData"
    test_jsons_dir   = "./test/jsons"
    generator = ImageMaskDatasetGenerator(test_images_dir, test_xmls_dir, 
                                          test_jsons_dir, output_dir,
                                          rename = False,
                                          resize = False,
                                          augmentation=False, debug=False)
    generator.generate()
    
  except:
    traceback.print_exc()