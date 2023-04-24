# Joshua Elledge
# 1001765744

import math
import random
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.spatial.distance import cdist
from matplotlib.patches import ConnectionPatch
from skimage.feature import SIFT, match_descriptors
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize, ProjectiveTransform, SimilarityTransform, warp, AffineTransform
from skimage import measure
from numpy.linalg import inv, pinv

def read_img_from_file(filename):
	img_rgb = np.asarray(Image.open(filename))
	if img_rgb.shape[2] == 4:
		img_rgb = rgba2rgb(img_rgb)
	return img_rgb

# Matches keypoints between two sets of descriptors using euclidean distance. Always cross-checks.
def keypoint_match(d1, d2):
	# Get euclidean distances between each d1 to each d2 
	distances = cdist(d1, d2, 'euclidean')

	matches1 = []
	for i in range(len(distances)):
		matches1.append(np.argmin(distances[i]))

	matches2 = []
	for i in range(len(distances[0])):
		matches2.append(np.argmin(distances[:, i]))

	matches_final = []
	for i in range(len(matches1)):
		proposed_match = matches1[i]
		if matches2[proposed_match] == i:
			matches_final.append([i, proposed_match])

	return np.asarray(matches_final)

def compute_affine_transform(kp_pairs):
	proj = []
	M = []

	for pair in kp_pairs:
		proj.append(pair[1][0])
		proj.append(pair[1][1])

		M.append([pair[0][0], pair[0][1], 1, 0, 0, 0])
		M.append([0, 0, 0, pair[0][0], pair[0][1], 1])

	proj = np.asarray(proj)
	M = np.asarray(M)

	trans = np.dot(pinv(M), proj)

	trans = np.array([
		[trans[0], trans[1], trans[2]],
		[trans[3], trans[4], trans[5]],
		[0, 0, 1]
	])

	return trans

def compute_projective_transform(kp_pairs):
	proj = []
	M = []

	for pair in kp_pairs:

		x1 = pair[0][0]
		x2 = pair[0][1]

		xh1 = pair[1][0]
		xh2 = pair[1][1]

		proj.append(xh1)
		proj.append(xh2)

		M.append([x1, x2, 1, 0, 0, 0, -x1 * xh1, -x2 * xh1])
		M.append([0, 0, 0, x1, x2, 1, -x1 * xh2, -x2 * xh2])

	proj = np.asarray(proj)
	M = np.asarray(M)

	trans = np.dot(pinv(M), proj)

	trans = np.array([
		[trans[0], trans[1], trans[2]],
		[trans[3], trans[4], trans[5]],
		[trans[6], trans[7], 1]
	])

	return trans

def apply_model(model, point):
	point = np.append(point, [1])
	perspective = point[2]
	point = np.dot(model, point)
	return point[:2] / perspective

def calc_residuals(transformed, destination):
	return np.sqrt(np.sum((transformed - destination) ** 2, axis = 1))

def RANSAC(src_kp, dst_kp, min_samples, threshold, max_trials, model_function):
	iterations = 0
	bestFit = None
	bestErr = np.inf
	bestInlierCount = 0
	bestInliers = []

	while iterations < max_trials:

		sample_indices = random.sample(range(src_kp.shape[0]), min_samples)
		maybeInliers = []
		for i in sample_indices:
			maybeInliers.append([src_kp[i], dst_kp[i]])

		maybeInliers = np.asarray(maybeInliers)
		maybeModel = model_function(maybeInliers)
		confirmedInliers = []

		residuals = calc_residuals(transform_points(maybeModel, src_kp), dst_kp)

		for i in range(src_kp.shape[0]):
			#print(residuals[i])
			if residuals[i] <= threshold:
				confirmedInliers.append([src_kp[i], dst_kp[i]])

		confirmedInliers = np.asarray(confirmedInliers)

		#print(confirmedInliers.shape)
		#print(maybeInliers.shape)
		if confirmedInliers.shape[0] > 0:
			betterModel = model_function(confirmedInliers)
			thisErr = np.average(calc_residuals(transform_points(betterModel, src_kp), dst_kp))
			if np.count_nonzero(confirmedInliers) > bestInlierCount or (np.count_nonzero(confirmedInliers) == bestInlierCount and thisErr < bestErr):
				bestFit = betterModel
				bestErr = thisErr
				bestInlierCount = np.count_nonzero(confirmedInliers)
				bestInliers = confirmedInliers

		iterations += 1

	return model_function(bestInliers), bestInliers

def transform_points(model, points):
	final = []
	for p in points:
		final.append(apply_model(model, p))
	return np.asarray(final)

def add_transforms(t1, t2):
	final = t1
	for i in range(len(t1)):
		for j in range(len(t1[0])):
			if not i == j:
				final[i][j] += t2[i][j]
	return final

dst_img_rgb = read_img_from_file('img/yosemite12.jpg')
src_img_rgb = read_img_from_file('img/yosemite34.jpg')

dst_img = rgb2gray(dst_img_rgb)
src_img = rgb2gray(src_img_rgb)

# KEYPOINT DETECTION
detector1 = SIFT()
detector2 = SIFT()
detector1.detect_and_extract(dst_img)
detector2.detect_and_extract(src_img)
keypoints1 = detector1.keypoints
descriptors1 = detector1.descriptors
keypoints2 = detector2.keypoints
descriptors2 = detector2.descriptors

# KEYPOINT MATCHING
matches = keypoint_match(descriptors1, descriptors2)

dst = keypoints1[matches[:, 0]]
src = keypoints2[matches[:, 1]]

# RANSAC
# Pass in compute_affine_transform function for affine, compute_projective_transform for projective
transform, best = RANSAC(src[:, ::-1], dst[:, ::-1], 4, 1, 300, compute_projective_transform)

src_best = best[:, 0]
dst_best = best[:, 1]

# COMPUTE THE OUTPUT SHAPE
rows, cols = dst_img.shape
corners = np.array([
	[0, 0],
	[cols, 0],
	[0, rows],
	[cols, rows]
])

# transform the corners of img1 by the inverse of the best fit model
corners_proj = transform_points(transform, corners)

all_corners = np.vstack((corners_proj[:, :2], corners[:, :2]))

corner_min = np.min(all_corners, axis=0)
corner_max = np.max(all_corners, axis=0)
output_shape = (corner_max - corner_min)
output_shape = np.ceil(output_shape[::-1]).astype(int)

offset = SimilarityTransform(translation = -corner_min)
dst_warped = warp(dst_img_rgb, inv(offset.params), output_shape=output_shape)

tf_img = warp(src_img_rgb, inv(add_transforms(transform, offset.params)), output_shape=output_shape)

# Combine the images
foreground_pixels = tf_img[tf_img > 0]
dst_warped[tf_img > 0] = tf_img[tf_img > 0]

# Plot the result
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.imshow(dst_warped)

plt.show()