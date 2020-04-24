//
//  main.cpp
//  hybridimg
//
//  Created by 任蕾 on 2020/3/23.
//  Copyright © 2020 任蕾. All rights reserved.
//

#include <iostream>
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;

void m_fft(string filename, Mat &dst) {
    Mat I = imread(filename, IMREAD_GRAYSCALE);
    if (I.empty()) {
        cout << "could not load image...%d\n" << endl;
        return ;
    }
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize(I.rows);
    int n = getOptimalDFTSize(I.cols); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, magI, 0, 1, NORM_MINMAX); // 归一化

    magI.copyTo(dst);
}

int main(int argc, const char * argv[]) {
    Mat src1, src2, dst1, dst2, src2_gray, dst;
    int sigma = 17;
    int ksize = (sigma*5)|1;
    char window_name[] = "hybridImg";
    src1 = imread("/Users/renlei/XcodeSpace/hybridimg/that2y.jpg", IMREAD_COLOR);
    if (src1.empty()) {
        cout << "could not load image1...%d\n" << endl;
        return -1;
    }
    src2 = imread("/Users/renlei/XcodeSpace/hybridimg/thaty.jpg", IMREAD_COLOR);
    if (src2.empty()) {
        cout << "could not load image2...%d\n" << endl;
        return -1;
    }
    GaussianBlur(src1, dst1, Size(ksize, ksize), sigma, sigma);
    imshow("Gaussian", dst1);
    imwrite("/Users/renlei/XcodeSpace/hybridimg/imgs/Gaussian.jpg", dst1);

    Mat tmp;
    GaussianBlur(src2, tmp, Size(ksize, ksize), sigma, sigma);
    // imshow("tmp", tmp);
    dst2 = src2 - tmp;
    // absdiff(src2, tmp, dst2);
    imshow("1-Gaussian", dst2);
    imwrite("/Users/renlei/XcodeSpace/hybridimg/imgs/1-Gaussian.jpg", dst2);
    
//    Mat tp;
//    Laplacian(tmp, tp, CV_16S, 5);
//    convertScaleAbs(tp, dst2, 1);
//    imshow("Laplacian", dst2);

//    int t = dst1.channels();
//    Mat tmp1 = Mat::zeros(dst1.size(), dst1.type());
//    double alp = 0.6;  // 对比度系数
//    double beta = 30;  // 亮度系数
//    for (int i=0; i<dst1.rows; i++) {
//        for (int j=0; j<dst1.cols; j++) {
//            if (t==3) {
//                float b = dst1.at<Vec3b>(i, j)[0];
//                float g = dst1.at<Vec3b>(i, j)[1];
//                float r = dst1.at<Vec3b>(i, j)[2];
//                tmp1.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(b*alp + beta);
//                tmp1.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(g*alp + beta);
//                tmp1.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(r*alp + beta);
//            }
//            else {
//                float gray = dst1.at<uchar>(i, j);
//                tmp1.at<uchar>(i, j) = saturate_cast<uchar>(gray*alp + beta);
//            }
//        }
//    }
    
    resize(dst2, dst2, Size(dst1.cols, dst1.rows));
    
    dst = dst1 + dst2;
    // imshow("tmp", dst);
    dst.convertTo(dst, -1, 0.8, 0);  // 对比度、亮度修饰
    imshow(window_name, dst);
    imwrite("/Users/renlei/XcodeSpace/hybridimg/imgs/hybrid.jpg", dst);
//    double alpha = 0.4;  // 图像的线性混合按照不同的权重进行混合dst1权重0.5，dst2权重0.5
//    addWeighted(dst1, alpha, dst2, (1-alpha), 3, dst);  // 图像混合API,权重相加
//    imshow(window_name, dst);
    Mat pyup, pydn;
    pyrUp(dst, pyup, Size(dst.cols*2, dst.rows*2));
    imshow("pyup1", pyup);
    imwrite("/Users/renlei/XcodeSpace/hybridimg/imgs/pyup1.jpg", pyup);
    pyrUp(pyup, pyup, Size(pyup.cols*2, pyup.rows*2));
    imshow("pyup2", pyup);
    imwrite("/Users/renlei/XcodeSpace/hybridimg/imgs/pyup2.jpg", pyup);
    pyrUp(pyup, pyup, Size(pyup.cols*2, pyup.rows*2));
    imshow("pyup3", pyup);
    imwrite("/Users/renlei/XcodeSpace/hybridimg/imgs/pyup3.jpg", pyup);
    pyrDown(dst, pydn, Size(dst.cols/2, dst.rows/2));
    imshow("pydn1", pydn);
    imwrite("/Users/renlei/XcodeSpace/hybridimg/imgs/pydn1.jpg", pydn);
    pyrDown(pydn, pydn, Size(pydn.cols/2, pydn.rows/2));
    imshow("pydn2", pydn);
    imwrite("/Users/renlei/XcodeSpace/hybridimg/imgs/pydn2.jpg", pydn);
    pyrDown(pydn, pydn, Size(pydn.cols/2, pydn.rows/2));
    imshow("pydn3", pydn);
    imwrite("/Users/renlei/XcodeSpace/hybridimg/imgs/pydn3.jpg", pydn);

    Mat fft1, fft2;
    m_fft("/Users/renlei/XcodeSpace/hybridimg/imgs/Gaussian.jpg", fft1);
    imshow("fft1", fft1);
    m_fft("/Users/renlei/XcodeSpace/hybridimg/imgs/1-Gaussian.jpg", fft2);
    imshow("fft2", fft2);
    
    waitKey(0);
     
    return 0;
}
