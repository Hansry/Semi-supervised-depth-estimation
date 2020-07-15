import cv2
import numpy as np
import matplotlib.pyplot as plt

cmap = plt.cm.viridis

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def convert_2d_to_3d(u, v, z, K):#将2d图像转到3d空间中
    v0 = K[1][2]
    u0 = K[0][2]
    fy = K[1][1]
    fx = K[0][0]
    x = (u-u0)*z/fx
    y = (v-v0)*z/fy
    return (x, y, z)

#特征点匹配
def feature_match(img1, img2, max_n_features):
   r''' Find features on both images and match them pairwise
   '''
   use_flann = False # better not use flann

   detector = cv2.xfeatures2d.SIFT_create(max_n_features)

   # find the keypoints and descriptors with SIFT
   kp1, des1 = detector.detectAndCompute(img1, None)
   kp2, des2 = detector.detectAndCompute(img2, None)
   if (des1 is None) or (des2 is None):
      return [], []
   des1 = des1.astype(np.float32)
   des2 = des2.astype(np.float32)

   if use_flann:
      # FLANN parameters
      FLANN_INDEX_KDTREE = 0
      index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
      search_params = dict(checks=50)
      flann = cv2.FlannBasedMatcher(index_params,search_params)
      matches = flann.knnMatch(des1,des2,k=2)
   else:
      matcher = cv2.DescriptorMatcher().create('BruteForce')
      matches = matcher.knnMatch(des1,des2,k=2)

   good = []
   pts1 = []
   pts2 = []
   # ratio test as per Lowe's paper
   for i,(m,n) in enumerate(matches):
      if m.distance < 0.8*n.distance:
         good.append(m)
         pts1.append(kp1[m.queryIdx].pt)
         pts2.append(kp2[m.trainIdx].pt)

   pts1 = np.int32(pts1)
   pts2 = np.int32(pts2)
   return pts1, pts2


def depth_colorize(depth):
   depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
   depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
   return depth.astype('uint8')

#使用F来剔除外点
def Fundamental(kp1,kp2):
    kp1=np.array(kp1)
    kp2=np.array(kp2)
    if(len(kp1)>=4):
       #H,mask=cv2.findHomography(kp1,kp2,cv2.RANSAC)
       F, mask = cv2.findFundamentalMat(kp1,kp2,cv2.FM_RANSAC,2,0.99)
       if F is None:
          print('F matrix is None.')
          return [],[]
       else:
          kp1=kp1[mask.ravel()==1]
          kp2=kp2[mask.ravel()==1]
    return kp1,kp2

##pnp求出位姿
def get_pose_pnp(rgb_curr, rgb_near, depth_curr, depth_near, K):#有这个oheight主要是因为对图片进行了底部裁剪，因此从3d->
   gray_curr = rgb2gray(rgb_curr).astype(np.uint8)
   gray_near = rgb2gray(rgb_near).astype(np.uint8)
   height, width = gray_curr.shape

   max_n_fetures_pose = 1000
   pts2d_curr, pts2d_near = feature_match(gray_curr, gray_near, max_n_fetures_pose)# feature matching
   #pts2d_curr, pts2d_near = Fundamental(pts2d_curr,pts2d_near)

   #对深度图进行膨胀
   kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4, 4))

   depth_curr_int = depth_curr.astype(np.int16)#需要将类型转成int16类型
   depth_near_int = depth_near.astype(np.int16)

   depth_curr_dilated = cv2.dilate(depth_curr_int, kernel)
   depth_near_dilated = cv2.dilate(depth_near_int, kernel)

   # extract 3d pts
   pts3d_curr = []
   pts2d_near_filtered = []  # keep only feature points with depth in the current frame

   pts3d_near = []
   pts2d_curr_filtered = []

   sparse_input = np.zeros((height, width))
   sparse_input_near = np.zeros((height, width))

#像素的横坐标u与纵坐标v分别是其图像数组中所在的列数与行数
   for i in range(len(pts2d_curr)): #提取pts2d_curr的特征点并给定深度

      u, v = pts2d_curr[i][0], pts2d_curr[i][1]#匹配上的特征点的个数
      u_n, v_n = pts2d_near[i][0], pts2d_near[i][1]

      z = depth_curr_dilated[v, u]
      z_n = depth_near_dilated[v_n, u_n]

      sparse_input[v, u] = z  #产生当前输入的稀疏深度图，有深度的信息即为特征点所在的位置
      sparse_input_near[v_n, u_n] = z_n #产生相邻帧的深度图


      if z > 0:
         xyz_curr = convert_2d_to_3d(u, v, z, K)
         pts3d_curr.append(xyz_curr)
         pts2d_near_filtered.append(pts2d_near[i])

      if z_n > 0:
         xyz_near = convert_2d_to_3d(u_n, v_n, z_n, K)
         pts3d_near.append(xyz_near)
         pts2d_curr_filtered.append(pts2d_curr[i])

   pts_for_loss = pts2d_curr
   pts_for_loss_near = pts2d_near

   # the minimal number of points accepted by solvePnP is 4:
   if len(pts3d_curr)>=4 and len(pts2d_near_filtered)>=4 and len(pts3d_near)>=4 and len(pts2d_curr_filtered)>=4:
      ##计算从curr到near的位姿
      pts3d_curr = np.expand_dims(np.array(pts3d_curr).astype(np.float32), axis=1)
      pts2d_near_filtered = np.expand_dims(np.array(pts2d_near_filtered).astype(np.float32), axis=1)

      # ransac
      ret = cv2.solvePnPRansac(pts3d_curr, pts2d_near_filtered, K, distCoeffs=None)
      success = ret[0]
      rotation_vector = ret[1]
      translation_vector = ret[2]

      ##计算从near到curr的位姿
      pts3d_near = np.expand_dims(np.array(pts3d_near).astype(np.float32), axis=1)
      pts2d_curr_filtered = np.expand_dims(np.array(pts2d_curr_filtered).astype(np.float32), axis=1)

      ret_inv = cv2.solvePnPRansac(pts3d_near, pts2d_curr_filtered, K, distCoeffs=None)
      success_inv = ret_inv[0]
      rotation_vector_inv = ret_inv[1]
      translation_vector_inv = ret_inv[2]
      return (success and success_inv, rotation_vector, translation_vector, rotation_vector_inv, translation_vector_inv,\
              sparse_input, depth_curr_dilated, sparse_input_near, depth_near_dilated, pts_for_loss, pts_for_loss_near)
   else:
      return (0, None, None, None, None, sparse_input, depth_curr_dilated, sparse_input_near, depth_near_dilated, pts_for_loss, pts_for_loss_near)