#include <stdio.h>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>

using namespace cv;
using namespace std;

Mat Padding(Mat img, int k_width, int k_height) //bordam matricea initiala cu 0-uri pt aplicarea ulterioara a kernelului
{
	Mat src;
	img.convertTo(src, CV_64FC1); //convertim imaginea initiala in greyscale
	int pad_rows, pad_cols;
	pad_rows = (k_height - 1) / 2;
	pad_cols = (k_width - 1) / 2;
	Mat pad_image(Size(src.cols + 2 * pad_cols, src.rows + 2 * pad_rows), CV_64FC1, Scalar(0)); //cream o matrice intermediara bordata cu 0-uri
	src.copyTo(pad_image(Rect(pad_cols, pad_rows, src.cols, src.rows))); //copiem matricea initiala in cea bordata

	return pad_image;
}

Mat Kernel(int k_width, int k_height) //crearea kernelului
{
	int pad_rows = (k_height - 1) / 2;
	int pad_cols = (k_width - 1) / 2;
	int sigma = 1;
	int K = 1;
	Mat kernel(k_height, k_width, CV_64FC1); //kernel greyscale

	for (int i = -pad_rows; i <= pad_rows; i++)
	{
		for (int j = -pad_cols; j <= pad_cols; j++)
		{
			kernel.at<double>(i + pad_rows, j + pad_cols) = K * exp(-(i * i + j * j) / (2.0 * sigma * sigma)); //cream kernelul dupa formula distributiei lui Gauss
		}
	}

	kernel = kernel / sum(kernel); //kernelul normalizat
	return kernel;
}

void Convolution(Mat src, Mat& dest, int k_width, int k_height)
{
	Mat pad_img, kernel;
	pad_img = Padding(src, k_width, k_height);
	kernel = Kernel(k_width, k_height);

	Mat output = Mat::zeros(src.size(), CV_64FC1);

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			output.at<double>(i, j) = sum(kernel.mul(pad_img(Rect(j, i, k_width, k_height)))).val[0]; //aplicarea kernelului pe imagine
		}
	}

	output.convertTo(dest, CV_8UC1);//imaginea finala in greyscaled dupa aplicarea filtrului Gauss
}

int main(int argc, char** argv)
{

	Mat image, dest3, dest5;
	image = imread("C:/Users/georg/Downloads/lenna.png", 0);
	if (!image.data)
	{
		printf("No image data \n");
		return -1;
	}
	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", image);
	waitKey(0);

	Convolution(image, dest3, 3, 3);
	namedWindow("3x3Gauss Image", WINDOW_AUTOSIZE);
	imshow("3x3Gauss Image", dest3);
	waitKey(0);

	Convolution(image, dest5, 5, 5);
	namedWindow("5x5Gauss Image", WINDOW_AUTOSIZE);
	imshow("5x5Gauss Image", dest5);
	waitKey(0);
	return 0;
}