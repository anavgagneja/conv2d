# agagneja_HW01

- Did not use zero padding so completed images are somewhat smaller. 
- Saving each output image to workspace directory as:
   ```
   Task<Task Number>_Image<First, Second, Third, etc. Kernel>_<Original Size>.jpg
   ```
- Input images stored in agagneja_HW01/images
	- Image1 is 1280 x 720
	- Image2 is 1920 x 1080

- Counted operations as total number of operations including kernel multiplication and addition as well as a divide step where I normalized my pictures by dividing by 3 * kernal_size ^ 2 for the 3 channels and each operation
- Program took too long so I was unable to include all pictures (especially for task 2 and the larger picture)
- Could not complete parts B and C because of how long program took to run
- test.py simply runs all 3 tasks on both images and outputs the files to 