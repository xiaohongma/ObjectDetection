#include "utils.h"

// ����matlab������Ӧ��ߵ���������
void _AdaptiveFindThreshold(CvMat *dx, CvMat *dy, double* low, double* high)
{
	CvSize size;
	IplImage *imge = 0;
	int i, j;
	CvHistogram *hist;
	int hist_size = 255;
	float range_0[] = { 0, 256 };
	float* ranges[] = { range_0 };
	double PercentOfPixelsNotEdges = 0.7;
	size = cvGetSize(dx);
	imge = cvCreateImage(size, IPL_DEPTH_32F, 1);
	// �����Ե��ǿ��, ������ͼ����
	float maxv = 0;
	for (i = 0; i < size.height; i++)
	{
		const short* _dx = (short*)(dx->data.ptr + dx->step*i);
		const short* _dy = (short*)(dy->data.ptr + dy->step*i);
		float* _image = (float *)(imge->imageData + imge->widthStep*i);
		for (j = 0; j < size.width; j++)
		{
			_image[j] = (float)(abs(_dx[j]) + abs(_dy[j]));
			maxv = maxv < _image[j] ? _image[j] : maxv; //�������ı�Եǿ��

		}
	}
	if (maxv == 0) {
		high = 0;
		low = 0;
		cvReleaseImage(&imge);
		return;
	}

	// ����ֱ��ͼ��bins��Χ�ǻҶȷ�Χ��ͳ����Ϊ�ݶ�ֵ��ѡ���ݶȱ���Ϊ0.7���ĻҶ�ֵ��Ϊ����ֵ��
	range_0[1] = maxv;
	hist_size = (int)(hist_size > maxv ? maxv : hist_size);//ѡ��bin�ķ�Χ������Ҷ�ֵ�ﲻ��255�����ý�Сֵ
	hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);
	cvCalcHist(&imge, hist, 0, NULL);
	int total = (int)(size.height * size.width * PercentOfPixelsNotEdges);
	float sum = 0;
	int icount = hist->mat.dim[0].size;
	//cout <<icount <<endl;

	float *h = (float*)cvPtr1D(hist->bins, 0);
	for (i = 0; i < icount; i++)
	{
		sum += h[i];
		if (sum > total)
			break;
	}
	// ����ߵ�����
	*high = (i + 1) * maxv / hist_size;
	*low = *high * 0.5;
	cvReleaseImage(&imge);
	cvReleaseHist(&hist);
}
void AdaptiveFindThreshold(const cv::Mat& image, double* low, double* high, int aperture_size)
{
	cv::Mat src = image;
	// imshow("Adaptive", src);
	const int cn = src.channels();
	cv::Mat dx(src.rows, src.cols, CV_16SC(cn));
	cv::Mat dy(src.rows, src.cols, CV_16SC(cn));

	cv::Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, cv::BORDER_REPLICATE);
	cv::Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0, cv::BORDER_REPLICATE);

	CvMat _dx = dx, _dy = dy;
	_AdaptiveFindThreshold(&_dx, &_dy, low, high);

}