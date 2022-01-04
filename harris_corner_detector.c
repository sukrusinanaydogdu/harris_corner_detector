#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include <cuda.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

__device__ char Sobel_Threshhold(const char Pixel_Value, const size_t Thresh)
{
	if (Pixel_Value >= Thresh)
		return char(255);
	else
		return char(0);
}

__global__  void Sobel_Operation(unsigned char *Input, unsigned char *Output, const int Width, const int Height, const size_t Thresh
									, int* dX2, int* dY2, int* dXY)
{
	//Variable for Gradient in X and Y direction and Final one
	float Gradient_h, Gradient_v;

	//Pixel value's
	char Pixel_Value = 0;

	//Calculating index id
	const unsigned int Col_Index = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int Row_Index = blockDim.y * blockIdx.y + threadIdx.y;

	if ((Row_Index < Height) && (Col_Index < Width))
	{
		if ((Row_Index != 0) && (Col_Index != 0) && (Row_Index != Height - 1) && (Col_Index != Width - 1))
		{
			Gradient_v = -(-Input[(Row_Index - 1)* Width + (Col_Index
				- 1)] + Input[(Row_Index - 1) * Width + (Col_Index + 1)] - 2 * Input[Row_Index * Width
				+ (Col_Index - 1)] + 2 * Input[Row_Index*Width + (Col_Index + 1)] - Input[(Row_Index + 1)*Width
				+ (Col_Index - 1)] + Input[(Row_Index + 1)*Width + (Col_Index + 1)]) / 8;

			Gradient_h = -(-Input[(Row_Index - 1)*Width + (Col_Index - 1)]
				- 2 * Input[(Row_Index - 1)*Width + Col_Index] - Input[(Row_Index - 1)*Width + (Col_Index + 1)]
				+ Input[(Row_Index + 1)*Width + (Col_Index - 1)] + 2 * Input[(Row_Index + 1)*Width + Col_Index]
				+ Input[(Row_Index + 1)*Width + (Col_Index + 1)]) / 8;

			//Assign Derivatives
			dX2[(Row_Index - 1) * Width + (Col_Index - 1)] = Gradient_h * Gradient_h;
			dY2[(Row_Index - 1) * Width + (Col_Index - 1)] = Gradient_v * Gradient_v;
			dXY[(Row_Index - 1) * Width + (Col_Index - 1)] = Gradient_h * Gradient_v;

			//Assign to image
			Pixel_Value = (char)sqrtf(Gradient_h * Gradient_h + Gradient_v * Gradient_v);
			Output[(Row_Index - 1) * Width + (Col_Index - 1)] = Sobel_Threshhold(Pixel_Value, Thresh);
		}

	}


}

__host__ __device__ void eigen_values(double M[2][2], double *l1,double *l2) 
{
	double d = M[0][0];
	double e = M[0][1];
	double f = M[1][0];
	double g = M[1][1];
	*l1 = ((d + g) + sqrt(pow(d + g, 2.0) - 4 * (d*g - f*e))) / 2.0f;
	*l2 = ((d + g) - sqrt(pow(d + g, 2.0) - 4 * (d*g - f*e))) / 2.0f;
}

__device__ double sum_neighbors(int *image, int row, int col, int cols, int window_dim)
{
	int window_center = window_dim / 2.0f;
	double sum = 0.0f;
	for (int i = 0; i<window_dim; ++i) {
		int image_row = (row - window_center) + i;
		for (int j = 0; j<window_dim; ++j) {
			int image_col = (col - window_center) + j;
			sum += (double)image[image_row * cols + image_col];
		}
	}
	return sum;
}
__global__ void detect_corners_kernel(int *dx2, int *dy2, int *dydx, int rows, int cols, double k, double *corner_response, int window_dim)
{
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int window_offset = window_dim / 2.0f;
	int image_row = ty;
	int image_col = tx;
	double M[2][2];

	if (image_row < rows - window_offset &&	image_col < cols - window_offset &&	image_row >= window_offset && image_col >= window_offset)
	{
		M[0][0] = sum_neighbors(dx2, image_row, image_col, cols, window_dim);
		M[0][1] = sum_neighbors(dydx, image_row, image_col,	cols, window_dim);
		M[1][1] = sum_neighbors(dy2, image_row, image_col, cols, window_dim);
		M[1][0] = M[0][1];
		double l1, l2;
		eigen_values(M, &l1, &l2);
		double r = l1 * l2 - k * pow(l1 + l2, 2.0);
		corner_response[image_row * cols + image_col] = r > 0 ? r : 0;
	}
}

__global__ void non_maxima_suppression_kernel(double *image, double *result, int rows, int cols, int window_dim) 
{
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int row = ty;
	int col = tx;
	int DIM = window_dim;
	int OFF = DIM / 2;
	if (row >= OFF && row < rows - OFF &&	col >= OFF && col < cols - OFF)
	{
		double filtered = image[row * cols + col];
		bool running = true;
		for (int i = 0; i<DIM && running; ++i) {
			int r = (row - OFF) + i;
			for (int j = 0; j<DIM && running; ++j) {
				int c = (col - OFF) + j;
				if (i == DIM / 2 && j == DIM / 2)
					continue;
				double temp = image[r * cols + c];
				if (temp > filtered) {
					filtered = 0;
					running = false;
				}
			}
		}
		result[row * cols + col] = filtered;
	}
}




// Function declarations.
unsigned char*
LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray);

void
WritePGM(char * sFileName, unsigned char * ucpDst_Host, int nWidth, int nHeight, int nMaxGray);

// Main function.
int
main(int argc, char ** argv)
{
	float fAverageTime = 0.0;

	for (int kk = 0; kk < 1; kk++)
	{	
		
		Mat img, img_gray, m_img;
		img = imread("lena_harris_input.png");
		cvtColor(img, img_gray, CV_BGR2GRAY);
		//imwrite("google.pgm", src);
		
		if (img_gray.rows > 100 || img_gray.cols > 100) {
			int newrows = 512;
			int newcols = img_gray.cols * newrows / img_gray.rows;

			resize(img_gray, img_gray, Size(newcols, newrows), 0, 0, INTER_CUBIC);
		}
		img_gray.copyTo(m_img);
		//imwrite("chess_512.pgm", m_img);
				

		// Host parameter declarations.	
		int   nWidth, nHeight, nMaxGray;

		// Load image to the host.
		size_t  size = 512 * 512;
		int length = 512 * 512;

		std::cout << "Load PGM file." << std::endl;
		unsigned char* h_buffer, *h_bufferResult, *d_buffer,*d_bufferResult, *d_bufferResultTemp;
		
		//Memory allocation for derivative terms
		int *h_XGradientSquared, *h_YGradientSquared,* d_XGradientSquared, *d_YGradientSquared, *h_XYDerivative, *d_XYDerivative;
		h_XGradientSquared = (int*)malloc(size * sizeof(int));
		h_YGradientSquared = (int*)malloc(size * sizeof(int));
		h_XYDerivative = (int*)malloc(size * sizeof(int));
		cudaMalloc((void**)&d_XGradientSquared, size * sizeof(int));
		cudaMalloc((void**)&d_YGradientSquared, size * sizeof(int));
		cudaMalloc((void**)&d_XYDerivative, size * sizeof(int));

		int* h_bufferINT, *h_bufferResultINT, *d_bufferINT, *d_bufferResultINT, *d_bufferResultTempINT;
		h_bufferINT = (int*)malloc(size * sizeof(int));
		h_bufferResultINT = (int*)malloc(size * sizeof(int));

		cudaMalloc((void**)&d_bufferINT, size * sizeof(int));
		cudaMalloc((void**)&d_bufferResultINT, size * sizeof(int));
		cudaMalloc((void**)&d_bufferResultTempINT, size * sizeof(int));


		h_buffer = (unsigned char*)malloc(size * sizeof(unsigned char));
		h_bufferResult = (unsigned char*)malloc(size * sizeof(unsigned char));

		cudaMalloc((void**)&d_buffer, size);
		cudaMalloc((void**)&d_bufferResult, size);
		cudaMalloc((void**)&d_bufferResultTemp, size);


		//LOAD IMAGE TO BUFFER
		//h_buffer = LoadPGM("myGray.pgm", nWidth, nHeight, nMaxGray);
		nWidth = 512;
		nHeight = 512;
		nMaxGray = 255;
		h_buffer = m_img.data;

		for (int i = 0; i < size; i++)
		{
			h_bufferINT[i] = (int)h_buffer[i];
		}
		
		int nImageSize = sizeof(unsigned char) * nHeight*nWidth;

		//TIMER STARTS
		float fElapsedTime;
		cudaEvent_t start, stop;			// Timer definition
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);			// Timer starts

		int nBlockSize = 16; // 16x16 blocks

		unsigned int unBlockRowNumber = (nHeight + nBlockSize - 1) / nBlockSize;
		unsigned int unBlockColumnNumber = (nWidth + nBlockSize - 1) / nBlockSize;
		dim3 dimGrid(unBlockColumnNumber, unBlockRowNumber);
		dim3 dimBlock(nBlockSize, nBlockSize);
		
		cudaMemcpy(d_buffer, h_buffer, nImageSize, cudaMemcpyHostToDevice);
		//CUDA method
		//dim3 block(16, 16);
		//dim3 grid(nWidth / 16, nHeight / 16);
		//d_blur << < grid, block >> >(d_buffer, d_bufferResult, nWidth, nHeight);

		//d_EdgeDetect << < dimGrid, dimBlock >> >(d_bufferINT, d_bufferResultINT, nWidth, nHeight);
		int sobel_threshold = 100;
		Sobel_Operation << < dimGrid, dimBlock >> >(d_buffer, d_bufferResult, nWidth, nHeight, sobel_threshold, d_XGradientSquared, d_YGradientSquared, d_XYDerivative);
	
		cudaDeviceSynchronize();
		cudaMemcpy(h_XGradientSquared, d_XGradientSquared, nImageSize * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_YGradientSquared, d_YGradientSquared, nImageSize * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_XYDerivative, d_XYDerivative, size*sizeof(int), cudaMemcpyDeviceToHost);
		
		//corner response parameters
		double k = 0.06; //sensitivity parameter
		int window_dim = 3;
		double *h_corner_response, *d_corner_response;
		h_corner_response = (double*)malloc(size * sizeof(double));
		cudaMalloc((void**)&d_corner_response, size * sizeof(double));

		//bu fonksiyondan corner_response'ı geri alacağız 
		detect_corners_kernel << < dimGrid, dimBlock >> > (d_XGradientSquared, d_YGradientSquared,d_XYDerivative, 
														nHeight, nWidth, k, d_corner_response, window_dim);
		
		cudaDeviceSynchronize();
		cudaMemcpy(h_corner_response, d_corner_response, size * sizeof(double), cudaMemcpyDeviceToHost);

		double *h_NMS_result, *d_NMS_result;
		h_NMS_result = (double*)malloc(size * sizeof(double));
		cudaMalloc((void**)&d_NMS_result, size * sizeof(double));

		//Non-maxima supression
		int NMS_window_dim = 3;
		non_maxima_suppression_kernel << < dimGrid, dimBlock >> >(d_corner_response, d_NMS_result, nHeight, nWidth, NMS_window_dim);
		cudaDeviceSynchronize();

		cudaMemcpy(h_corner_response, d_NMS_result, size * sizeof(double), cudaMemcpyDeviceToHost);

		/*
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; ++j) {
				if (suppressed[i * cols + j] > 0.0) {
					features.push_back(cv::KeyPoint(j, i, 5, -1));
				}
			}
		}
		*/

			
		//d_blur << < dimGrid, dimBlock >> >(d_bufferINT, d_bufferResultINT, nWidth, nHeight);

		//convolve_kernel_seperable_horizontal<unsigned char> << < dimGrid, dimBlock>> >(d_buffer, d_bufferResultTemp, nHeight, nWidth, -1, 0, 1);
		//cudaDeviceSynchronize();
		//convolve_kernel_seperable_vertical<unsigned char> << < dimGrid,dimBlock>> >(d_bufferResultTemp, d_bufferResult, nHeight, nWidth, 1, 2, 1);

		cudaEventRecord(stop, 0);								// Timer stops
		cudaEventSynchronize(stop);								// Gives synchronous processing between threads
		cudaEventElapsedTime(&fElapsedTime, start, stop);		// Takes the elapsed time spent in the kernel
		cudaEventDestroy(start);
		cudaEventDestroy(stop);									// Deallocate the timer events

		fAverageTime += fElapsedTime;

		cudaMemcpy(h_bufferResult, d_bufferResult, nImageSize,cudaMemcpyDeviceToHost);
		
		
		for (int i = 0; i < size; i++)
		{
			h_bufferResult[i] = (unsigned char)((int)h_corner_response[i]);

		}

		
		
		// Output the result image.
		printf("\n");
		std::cout << "Output the PGM file." << std::endl;
		WritePGM("lena_harris_result.pgm", h_bufferResult, nWidth, nHeight, nMaxGray);


		// Clean up.
		printf("\n");
		std::cout << "Clean up." << std::endl;
		//free(h_buffer);
		free(h_bufferResult);

	}

	fAverageTime = fAverageTime / 1;		// Average time after 10 iterations is obtained and printed
	printf("\nElapsed Time = %f \r\n", fAverageTime);

	system("pause");
	return 0;
}

// Disable reporting warnings on functions that were marked with deprecated.
#pragma warning( disable : 4996 )

// Load PGM file.
unsigned char *
LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray)
{
	char aLine[256];
	FILE * fInput = fopen(sFileName, "r");
	if (fInput == 0)
	{
		perror("Cannot open file to read");
		exit(EXIT_FAILURE);
	}
	// First line: version
	fgets(aLine, 256, fInput);
	std::cout << "\tVersion: " << aLine;
	// Second line: comment
	fgets(aLine, 256, fInput);
	std::cout << "\tComment: " << aLine;
	fseek(fInput, -1, SEEK_CUR);
	// Third line: size
	fscanf(fInput, "%d", &nWidth);
	std::cout << "\tWidth: " << nWidth;
	fscanf(fInput, "%d", &nHeight);
	std::cout << " Height: " << nHeight << std::endl;
	// Fourth line: max value
	fscanf(fInput, "%d", &nMaxGray);
	std::cout << "\tMax value: " << nMaxGray << std::endl;
	while (getc(fInput) != '\n');
	// Following lines: data
	unsigned char * ucpSrc_Host = new unsigned char[nWidth * nHeight];
	for (int i = 0; i < nHeight; ++i)
		for (int j = 0; j < nWidth; ++j)
			ucpSrc_Host[i*nWidth + j] = fgetc(fInput);
	fclose(fInput);

	return ucpSrc_Host;
}

// Write PGM image.
void
WritePGM(char * sFileName, unsigned char * ucpDst_Host, int nWidth, int nHeight, int nMaxGray)
{
	FILE * fOutput = fopen(sFileName, "w+");
	if (fOutput == 0)
	{
		perror("Cannot open file to read");
		exit(EXIT_FAILURE);
	}
	char * aComment = "# Created by GPU";
	fprintf(fOutput, "P5\n%s\n%d %d\n%d\n", aComment, nWidth, nHeight, nMaxGray);
	for (int i = 0; i < nHeight; ++i)
		for (int j = 0; j < nWidth; ++j)
			fputc(ucpDst_Host[i*nWidth + j], fOutput);
	fclose(fOutput);
}
