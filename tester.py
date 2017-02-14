import matplotlib.pyplot as plt
exec(open("./data_augmentation.py").read())
exec(open("./model.py").read())


# def test_pick_random_image_idx():
# 	random_idx_gen = pick_random_image_idx(csv_data) 
# 	rows_test = []
# 	camera_test = []
# 	for i in range(100000):
# 		row_i, camera_i = next(random_idx_gen)
# 		rows_test.append(row_i)
# 		camera_test.append(camera_i)

# 	rows,row_counts = np.unique(rows_test, return_counts = True)
# 	cameras, camera_counts = np.unique(camera_test, return_counts = True)
# 	pdb.set_trace()


def figplot(img1, img2):
	plt.figure(figsize=(30,30))
	plt.subplot(1,2,1)
	plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
	plt.xticks([])
	plt.yticks([])
	plt.subplot(1,2,2)
	plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
	# plt.xticks([])
	# plt.yticks([])	
	plt.show(block=False)

image_1 = cv2.imread('./IMG/center_2016_12_01_13_30_48_287.jpg')
# image_1 = img_to_array(load_img('./IMG/center_2016_12_01_13_30_48_287.jpg'))
def test_trans_image(image_1):
	image_1 = cv2.imread('./IMG/center_2016_12_01_13_30_48_287.jpg')
	image_2, shift_amt= trans_image(image_1, (20,20),return_rand_param = True)
	print(shift_amt)
	figplot(image_1,image_2)

def test_augment_image_brightness(image_1):
	image_2, bright_mag= augment_image_brightness(image_1,return_rand_param = True)
	print(bright_mag)
	figplot(image_1,image_2)

def test_add_random_shadow(image_1):
	image_2= add_random_shadow(image_1)
	figplot(image_1,image_2)

def test_hflip(image_1):
	image_2, flipped= hflip(image_1, return_rand_param = True)
	print(flipped)
	figplot(image_1,image_2)

def test_vflip(image_1):
	image_2, flipped= vflip(image_1, return_rand_param = True)
	print(flipped)
	figplot(image_1,image_2)

def test_rotate_image(image_1):
	image_2, rot_amt= rotate_image(image_1, (-10,10),return_rand_param = True)
	print(rot_amt)
	figplot(image_1,image_2)

def test_shear_image(image_1, shear_range = (-10,10)):
	image_2, shear_amt= shear_image(image_1,shear_range =shear_range,return_rand_param = True)
	print(shear_amt)
	figplot(image_1,image_2)

def test_augment_image_nomod(image_1):
	image_2, AugImg_Values= augment_image(image_1,return_rand_param = True)
	print(AugImg_Values)
	figplot(image_1,image_2)

def test_augment_image(image_1):
	image_2, AugImg_Values = augment_image(image_1, shear_range = (0,0), rot_range =(0,0),
	                                            vflip_prob = 0.5, hflip_prob = 0, add_shadow = True,shadow_intensity = 0.9,
	                                            trans_range = (20,20), brightness_range = (0.25,1.5), return_rand_param = True)
	figplot(image_1,image_2)

def test_my_train_datagen():
	csv_data = pd.read_csv('./driving_log.csv')
	my_gen = my_train_datagen(csv_data,(1,2), batch_size = 1)
	(img, steering) = next(my_gen)
	# show
	plt.figure(figsize=(5,5))
	# plt.imshow(cv2.cvtColor(img.squeeze().astype('uint8'), cv2.COLOR_BGR2RGB))
	plt.imshow(img.squeeze().astype('uint8'))
	plt.xticks([])
	plt.yticks([])
	plt.show(block=False)
	return img


