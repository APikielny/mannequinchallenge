#for reordering frames

path = '/home/adam/Desktop/repos/mannequinchallenge/test_data/custom-handheld/translate/frame'
file1 = open("test_data/test_marc_video_list_reorder.txt", "a")
num_frames = 75
for i in range(num_frames):
    file1.write(path + str(i).zfill(4) + ".jpg\n") #pad with zeroes for sorting later
file1.close()