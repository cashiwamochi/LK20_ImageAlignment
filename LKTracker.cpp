#include "LKTracker.hpp"

using namespace std;

LKTracker::LKTracker(cv::Mat& image, CalcType t)
: mnHeight(image.cols), mnWidth(image.rows), mType(t)
{
	mmRefImage = image.clone();

	cv::Mat gray;
	if(image.channels() == 3) {
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	}
	else {
		gray = image.clone();
	}

	gray.convertTo(mmRefWorkingImage, CV_32FC1, 1.0/255.0);

	mITER_MAX = 50;

	RegisterSL3();
	if(mType == LKTracker::IC) {
		PreComputeIC();
	}
	else if(mType == LKTracker::FC) {
		mvmSteepestDescentImage.clear();
		mmHessian.release();
		mvfResidual.clear();
		PreComputeFC();
	}
	else if(mType == ESM) {
		PreComputeESM();
	}

	cout << "[MODE] ";
	if(mType == FC) {
		cout << "FORWARD COMPOSITIONAL\n";
	}
	else if(mType == IC) {
		cout << "INVERSE COMPOSITIONAL\n";
	}
	else if(mType == ESM) {
		cout << "Efficient Second-Order Minimization\n";
	}

}

void LKTracker::SetInitialWarp(cv::Mat& _H0) {
	mmW0 = _H0.clone();
}

cv::Size LKTracker::GetTrackSize()
{
	return cv::Size(mnWidth,mnHeight);
}

void LKTracker::RegisterSL3() {
	/* PAPER p.6 sl3 bases matrices */
	// Register SL3 bases
	mvmSL3Bases.resize(8);
	mvmSL3Bases[0] = (cv::Mat_<float>(3,3) << 0.0, 0.0, 1.0,
																						0.0, 0.0, 0.0,
																						0.0, 0.0, 0.0);
	mvmSL3Bases[1] = (cv::Mat_<float>(3,3) << 0.0, 0.0, 0.0,
																						0.0, 0.0, 1.0,
																						0.0, 0.0, 0.0);
	mvmSL3Bases[2] = (cv::Mat_<float>(3,3) << 0.0, 1.0, 0.0,
																						0.0, 0.0, 0.0,
																						0.0, 0.0, 0.0);
	mvmSL3Bases[3] = (cv::Mat_<float>(3,3) << 0.0, 0.0, 0.0,
																						1.0, 0.0, 0.0,
																						0.0, 0.0, 0.0);
	mvmSL3Bases[4] = (cv::Mat_<float>(3,3) << 1.0, 0.0, 0.0,
																						0.0,-1.0, 0.0,
																						0.0, 0.0, 0.0);
	mvmSL3Bases[5] = (cv::Mat_<float>(3,3) << 0.0, 0.0, 0.0,
																						0.0,-1.0, 0.0,
																						0.0, 0.0, 1.0);
	mvmSL3Bases[6] = (cv::Mat_<float>(3,3) << 0.0, 0.0, 0.0,
																						0.0, 0.0, 0.0,
																						1.0, 0.0, 0.0);
	mvmSL3Bases[7] = (cv::Mat_<float>(3,3) << 0.0, 0.0, 0.0,
																						0.0, 0.0, 0.0,
																						0.0, 1.0, 0.0);

	/* PAPER p.12 (65) */
 	// make Jg
 	mmJg = cv::Mat::zeros(9, 8, CV_32FC1);
	// SL3の微分は各基底を横に並べて転置をとったベクトルを横に並べたもの
	for(int i = 0; i < 8; i++) {
	 	for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++) {
				mmJg.at<float>(j*3 + k,i) = mvmSL3Bases[i].at<float>(j,k);
		 	}
	 	}
	}

	return;
}

float LKTracker::ComputeResidual() {
	float fResidual = 0.f;

	mvfResidual.clear();
	mvfResidual.reserve(mnHeight*mnWidth);
	for(int v = 0; v < mnHeight; v++) {
		for(int u = 0; u < mnWidth; u++) {
			float r;
			if(mType == IC or mType == ESM) {
				r = mmCurrentWorkingImage.at<float>(v,u) - mmRefWorkingImage.at<float>(v,u);
			}
			else if(mType == FC) {
				r = mmRefWorkingImage.at<float>(v,u) - mmCurrentWorkingImage.at<float>(v,u);
			}
			mvfResidual.push_back(r);
			fResidual += r*r;
		}
	}
	return sqrt(fResidual/(mnHeight*mnWidth));
}

void LKTracker::PreComputeIC() {
	// Evaluate the gradient of the template
	mmRefImageDx.create(mnHeight*mnWidth,1,CV_32F);
	mmRefImageDy.create(mnHeight*mnWidth,1,CV_32F);

	for(int v = 0; v < mnHeight; v++) {
		for(int u = 0; u < mnWidth; u++) {
			const int idx = u + v*mnWidth;

			// dx-image
			if( u+1 == mnWidth or v+1 == mnHeight ) {
				mmRefImageDx.at<float>(idx) = 0.f;
			}
			else {
			 	mmRefImageDx.at<float>(idx) = mmRefWorkingImage.at<float>(v,u+1) - mmRefWorkingImage.at<float>(v,u);
			}

			// //dy-image
			if( u+1 == mnWidth or v+1 == mnHeight ) {
			 	mmRefImageDy.at<float>(idx) = 0.f;
			 }
			 else {
			 	mmRefImageDy.at<float>(idx) = mmRefWorkingImage.at<float>(v+1,u) - mmRefWorkingImage.at<float>(v,u);
			 }
		}
	}

	// dW/dp (p=0,W=I)
	mvmfJw.clear();
	mvmfJw.reserve(mnHeight * mnWidth);
	for(int _v = 0; _v < mnHeight; _v++) {
		for(int _u = 0; _u < mnWidth; _u++) {
			float u = (float)_u;
			float v = (float)_v;

			/* PAPER p.12 (63) ただし，第3行成分は実際の計算では0で意味がないので削った */
			cv::Mat _mJw = (cv::Mat_<float>(2,9) <<
			u, v, 1.f, 0.f, 0.f, 0.f, -u*u, -u*v, -u,
			0.f, 0.f, 0.f, u, v, 1.f, -u*v, -v*v, -v);

			cv::Mat mfjwjg = _mJw * mmJg;
			// [2x8] = [2x9] x [9x8]
			mvmfJw.push_back(mfjwjg);

		}
	}

	int n_params = 8;
	mmHessian = cv::Mat::zeros(n_params, n_params, CV_32F);

	mvmSteepestDescentImage.reserve(mnHeight*mnWidth);
	// Compute ComputeSteepestDescentImage
	for(int v = 0; v < mnHeight; v++) {
		for(int u = 0; u < mnWidth; u++) {
			const int idx = u + v*mnWidth;
			cv::Mat mDxDy = (cv::Mat_<float>(1,2) << mmRefImageDx.at<float>(idx), mmRefImageDy.at<float>(idx));
			cv::Mat J = mDxDy * mvmfJw[idx]; // [1x8] = [1x2] x [2x8]
			mvmSteepestDescentImage.push_back(J);
			mmHessian += J.t() * J; // [8x8] = [8x1] x [1x8]. DoF is 8
		}
	}

	return;
}


void LKTracker::PreComputeFC() {
	// dW/dp (p=0,W=I)
	mvmfJw.clear();
	mvmfJw.reserve(mnHeight * mnWidth);
	for(int _v = 0; _v < mnHeight; _v++) {
		for(int _u = 0; _u < mnWidth; _u++) {
			float u = (float)_u;
			float v = (float)_v;

			/* PAPER p.12 (63) ただし，第3行成分は実際の計算では0で意味がないので削った */
			cv::Mat _mJw = (cv::Mat_<float>(2,9) <<
			u, v, 1.f, 0.f, 0.f, 0.f, -u*u, -u*v, -u,
			0.f, 0.f, 0.f, u, v, 1.f, -u*v, -v*v, -v);

			cv::Mat mfjwjg = _mJw * mmJg;
			// [2x8] = [2x9] x [9x8]
			mvmfJw.push_back(mfjwjg);
		}
	}
	return;
}


void LKTracker::PreComputeESM() {
	PreComputeIC();
	return;
}


// This is used in FC or ESM
void LKTracker::ComputeHessian(cv::Mat& mHessian, std::vector<cv::Mat>& mvmSteepestDescentImage) {
	// Evaluate the gradient of the template
	mHessian.release();
	mvmSteepestDescentImage.clear();

	const int n_params = 8;
	mmHessian = cv::Mat::zeros(n_params, n_params, CV_32F);

	float grad_x = 0.0;
	float grad_y = 0.0;
	for(int v = 0; v < mnHeight; v++) {
		for(int u = 0; u < mnWidth; u++) {
			const int idx = u + v*mnWidth;

			// dx-image
			if( u+1 == mnWidth or v+1 == mnHeight ) {
				grad_x = 0.f;
			}
			else {
				grad_x = mmCurrentWorkingImage.at<float>(v,u+1) - mmCurrentWorkingImage.at<float>(v,u);
			}

			// dy-image
			if( u+1 == mnWidth or v+1 == mnHeight ) {
				grad_y = 0.f;
			}
			else {
				grad_y = mmCurrentWorkingImage.at<float>(v+1,u) - mmCurrentWorkingImage.at<float>(v,u);
			}
		
			cv::Mat mDxDy;
			if(mType == FC) {
				mDxDy = (cv::Mat_<float>(1,2) << grad_x, grad_y);
			}
			else if(mType == ESM) {
				mDxDy = (cv::Mat_<float>(1,2) << (grad_x + mmRefImageDx.at<float>(idx))/2.0, 
																				 (grad_y + mmRefImageDy.at<float>(idx))/2.0);
			}
			cv::Mat J = mDxDy * mvmfJw[idx]; // [1x8] = [1x2] x [2x8]
			mvmSteepestDescentImage.push_back(J);
			mHessian += J.t() * J; // [8x8] = [8x1] x [1x8]. DoF is 8
		}
	}
	return;
}

vector<float> LKTracker::ComputeUpdateParams() {
	// changed here
	int n_params = 8;
	vector<float> vfParams(n_params, 0.f);
	cv::Mat mfParams = cv::Mat::zeros(n_params, 1, CV_32F);
	for(int v = 0; v < mnHeight; v++) {
		for(int u = 0; u < mnWidth; u++) {
			const int idx = u + v*mnWidth;
			cv::Mat _p;
			if(mType == ESM) {
				_p = -1.0*mmHessian.inv() * mvmSteepestDescentImage[idx].t() * mvfResidual[idx];
			}
			else {
				_p = mmHessian.inv() * mvmSteepestDescentImage[idx].t() * mvfResidual[idx];
			}
			// [8*1] = [8x8] * [8*1] * [1]

			mfParams += _p; // 8
		}
	}

	for(int i = 0; i < n_params; i++) {
		vfParams[i] = mfParams.at<float>(i);
	}

	return vfParams;
}

void LKTracker::UpdateWarp(const std::vector<float> vfParams, cv::Mat &W){

	/* PAPER (41) (42) */
	cv::Mat A = cv::Mat::zeros(3,3,CV_32F);
	for(int i = 0; i < 8; i++) {
		A += (vfParams[i] * mvmSL3Bases[i]);
	}

	cv::Mat G = cv::Mat::zeros(3, 3, CV_32F);
	cv::Mat A_i = cv::Mat::eye(3, 3, CV_32F);
	float factor_i = 1.f;

	for(int i = 0; i < 10; i++) {
		G += (1.0/factor_i)*(A_i);
		A_i *= A;
		factor_i *= (float)(i+1);
	}

	cv::Mat deltaW = G.clone();

  // update!!
	if(mType == IC) {
		W = W * deltaW.inv();
	}
	else if(mType == FC or mType == ESM) {
		W = W * deltaW;
	}

	return;
}

// main
bool LKTracker::Track(const cv::Mat& target_image, cv::Mat& H, cv::Mat& dst_image) {
	// pre-process
	mmCurrentImage.release();
	cv::Mat gray;
	if(target_image.channels() == 3) {
		cv::cvtColor(target_image, gray, cv::COLOR_BGR2GRAY);
	}
	else {
		gray = target_image.clone();
	}
	mmCurrentImage = gray.clone();
	gray.convertTo(mmCurrentWorkingImage, CV_32FC1, 1.0/255.0);
	mmOriginalWorkingImage = mmCurrentWorkingImage.clone();

	if(mType == FC) {
		// ICでは一度しか計算されないがFCでは毎回更新される
		assert(mvmSteepestDescentImage.empty());
		assert(mvfResidual.empty());
		assert(mmHessian.empty());
	}

	float previous_error = -1.0;
	cv::Mat mPrevW = cv::Mat::eye(3,3,CV_32F);
	cv::Mat W = mPrevW.clone();

	for(int itr = 0; itr < mITER_MAX ; itr++) {
		if(mType == LKTracker::IC) {
			// Inverse Compositional Algorithm
			WarpCurImg(W);
			float current_error = ComputeResidual();
			fprintf(stderr,"Itr[%d]: %.6f\n",itr,current_error);
			vector<float> vfUpdateParam = ComputeUpdateParams();
			cv::Mat newW = W.clone();
			UpdateWarp(vfUpdateParam, newW);
			if(!IsConverged(current_error, previous_error)) 
			{	
				previous_error = current_error;
				W = newW.clone();
			}
			else {
				break;
			}
		}
		else if(mType == LKTracker::FC) {
			// Forward Compositional Algorithm
			WarpCurImg(W);
			float current_error = ComputeResidual();
			fprintf(stderr,"Itr[%d]: %.6f\n",itr,current_error);
			ComputeHessian(mmHessian, mvmSteepestDescentImage);
			vector<float> vfUpdateParam = ComputeUpdateParams();
			cv::Mat newW = W.clone();
			UpdateWarp(vfUpdateParam, newW);

			if(!IsConverged(current_error, previous_error)) 
			{
				previous_error = current_error;
				W = newW.clone();
			}
			else {
				break;
			}	
		}
		else if(mType == LKTracker::ESM) {
			// Efficient Second-Order Minization
			WarpCurImg(W);
			float current_error = ComputeResidual();
			fprintf(stderr,"Itr[%d]: %.6f\n",itr,current_error);
			ComputeHessian(mmHessian, mvmSteepestDescentImage);
			vector<float> vfUpdateParam = ComputeUpdateParams();
			cv::Mat newW = W.clone();
			UpdateWarp(vfUpdateParam, newW);

			if(!IsConverged(current_error, previous_error)) 
			{
				previous_error = current_error;
				W = newW.clone();
			}
			else {
				break;
			}
		}
	}

	H = W.clone();
	mPrevW = W.clone();
	mmCurrentWorkingImage.convertTo(dst_image, CV_8UC1, 255);
	return true;
}

void LKTracker::WarpCurImg(const cv::Mat &W) {
	cv::warpPerspective(mmOriginalWorkingImage, mmCurrentWorkingImage, mmW0*W,
					cv::Size(mnWidth, mnHeight),
					cv::INTER_LINEAR + cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT,
					cv::Scalar(0,0,0));
	return;
}

bool LKTracker::IsConverged(const float& current_error, const float& previous_error) {
	bool b_stop_iteration = false;
	if(previous_error < 0.0) {
		return false;
	}
	if(previous_error <= current_error + 0.0000001) {
		b_stop_iteration = true;
	}
	return b_stop_iteration;
}
