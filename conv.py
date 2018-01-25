import numpy as numpy
import torch 
from torchvision.utils import save_image as save

class Conv2D:
    def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode

    def assign_kernels(self):
        # Define kernels k1-k5
        kernel_k1 = torch.FloatTensor([[-1, -1, -1],  [0, 0, 0],  [1, 1, 1]])
        kernel_k2 = torch.FloatTensor([[-1,  0,  1], [-1, 0, 1], [-1, 0, 1]])
        kernel_k3 = torch.FloatTensor([[ 1,  1,  1], [1, 1, 1],  [1, 1, 1]])
        kernel_k4 = torch.FloatTensor([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        kernel_k5 = torch.FloatTensor([[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1]])

        # Assign list of kernels based on mode, o_channel, kernel_size
        kernels = []
        if self.mode == 'known':
            if self.o_channel == 1 and self.kernel_size == 3:
                kernels.append(kernel_k1)
            elif self.o_channel == 2 and self.kernel_size == 5:
                kernels.append(kernel_k4)
                kernels.append(kernel_k5)
            elif self.o_channel == 3 and self.kernel_size == 3:
                kernels.append(kernel_k1)
                kernels.append(kernel_k2)
                kernels.append(kernel_k3)
            else: 
                print('Error: Unkown Kernel Choices. Please select different o_channel and kernel_size values or select rand mode.')
                return
        elif self.mode == 'rand':
            for x in range(self.o_channel):
                kernels.append(torch.rand(self.kernel_size, self.kernel_size))
        else:
            print('Error: Please select known or rand for mode')
            return

        inverted_kernels = []
        for kernel in kernels:
            # Place inverted version of kernels in inverted_kernels
            kernel_flip = numpy.fliplr(kernel.numpy())
            kernel_flip = numpy.flipud(kernel_flip)
            inverted_kernels.append(torch.from_numpy(kernel_flip.copy()))
        return kernels, inverted_kernels

    def convolve_small(self, kernel, input_image):
        rows = len(input_image[0])
        cols = len(input_image[0][0])
        conv_rows = ((rows - self.kernel_size) // self.stride) + 1
        conv_cols = ((cols - self.kernel_size) // self.stride) + 1
        result_rgb = torch.zeros(conv_rows, conv_cols)
        normalize = 3 * self.kernel_size * self.kernel_size
        operation_count = 0
        for channel in range(3):
            result = torch.zeros(conv_rows, conv_cols)
            iCount = 0
            for i in range(0, conv_rows * self.stride, self.stride):
                jCount = 0
                for j in range(0, conv_cols * self.stride, self.stride):
                    for m in range(i, self.kernel_size + i):
                        for n in range(j, self.kernel_size + j):
                            result[iCount][jCount] += input_image[channel][m][n] * kernel[m - i][n - j]
                            operation_count += 2
                    jCount+= 1
                iCount += 1
            result_rgb = torch.add(result, result_rgb)
            result_rgb_div = torch.div(result_rgb, normalize)
        operation_count += (conv_cols * conv_rows * 3)
        return result_rgb, result_rgb_div, operation_count

    def forward(self, input_image):
        # Transpose input_image so that dimensions are (Channel x Height x Width)
        input_image = torch.transpose(input_image, 0, 2)
        input_image = torch.transpose(input_image, 1, 2)

        # Get kernels and inverted kernels given o_channel, kernel_size, and mode
        kernels, inverted_kernels = self.assign_kernels()
        rows = len(input_image[0])
        cols = len(input_image[0][0])
        conv_rows = ((rows - self.kernel_size) // self.stride) + 1
        conv_cols = ((cols - self.kernel_size) // self.stride) + 1
        solution = torch.zeros(self.o_channel, conv_rows, conv_cols)
        total_operations = 0
        for i in range(len(inverted_kernels)):
            result, result_div, number_operations = self.convolve_small(inverted_kernels[i], input_image)
            solution[i] = result_div
            total_operations += number_operations
            save(solution[i], 'Task ' + str(self.o_channel) + '_' + 'Image' + str(i + 1) + '_' + str(cols) + 'x' + str(rows) + '.jpg') 
        return total_operations, solution
        