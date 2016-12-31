#include <opencv/cv.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <string.h>
#include <fstream>
#include <random>

using namespace cv;
using namespace std;

bool openImage(const std::string &filename, Mat &image);
void train(vector<string> &listOfImages);
void bagOfWords(vector<string> &listOfImages, vector<string> &listOfTestImages);
void read_csv(const string& filename, vector<string> label);
void applySVM(vector<string> &listOfTestImages, BOWImgDescriptorExtractor bowExtractor);
Mat allTrainDescriptors, bagofWords, labels;
//the number of bags
int dictionarySize = 150;
//define Term Criteria
TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
vector<int> emptydescriptors;
vector<string> label;



int main(int argc, char** argv)
{
	initModule_nonfree();
	/*
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
	*/
	///////////////////////////////////////

	vector<string> listOfImages;
	for (int i = 0; i < 50000; i++)
		listOfImages.push_back(to_string(i+1) + ".png");
	vector<string> listOfTestImages;
	for (int i = 0; i < 300000; i++)
		listOfTestImages.push_back(to_string(i + 1) + ".png");
	
	train(listOfImages);

	label.push_back("airplane");
	label.push_back("automobile");
	label.push_back("bird");
	label.push_back("cat");
	label.push_back("deer");
	label.push_back("dog");
	label.push_back("frog");
	label.push_back("horse");
	label.push_back("ship");
	label.push_back("truck");


	bagOfWords(listOfImages, listOfTestImages);


	//applySVM(listOfTestImages);

	waitKey(0);
	return 0;
}

void read_csv(const string& filename, vector<string> label) {
	ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	// For each line in the given file:
	getline(file, line);
	while (getline(file, line)) {
		// Get the current line:
		stringstream liness(line);
		// Split it at the colon:
		getline(liness, path, ',');
		//cout << path << endl;
		getline(liness, classlabel);
		//cout << classlabel << endl;
		//labels.push_back(atoi(classlabel.c_str()));
		if (find(emptydescriptors.begin(), emptydescriptors.end(), atoi(path.c_str())) != emptydescriptors.end()) {
			continue;
		}
		else {
			int pos = find(label.begin(), label.end(), classlabel) - label.begin();
			//cout << pos << endl;
			labels.push_back(pos);
		}
	}
}

bool openImage(const std::string &filename, Mat &image)
{
	image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (!image.data) {
		std::cout << " --(!) Error reading image " << filename << std::endl;
		return false;
	}
	return true;
}

void train(vector<string> &listOfImages)
{
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
	Mat image;
	std::vector<KeyPoint> keypoints;
	Mat descriptors;

	for (int i = 0; i<listOfImages.size(); i++)
	{
		if (!openImage("train/" + listOfImages[i], image))
			continue;
		cout << "Extracting from Image: " + listOfImages[i]+ " of "+to_string(listOfImages.size()) << endl;
		detector->detect(image, keypoints);
		extractor->compute(image, keypoints, descriptors);
		allTrainDescriptors.push_back(descriptors);
	}
}

void bagOfWords(vector<string> &listOfImages, vector<string> &listOfTestImages) {
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
	Mat image;
	std::vector<KeyPoint> keypoints;
	Mat descriptors;

	BOWKMeansTrainer bowTrainer(dictionarySize, tc, 1, KMEANS_PP_CENTERS);
	BOWImgDescriptorExtractor bowExtractor(extractor, matcher);
	Mat dictionary = bowTrainer.cluster(allTrainDescriptors);
	bowExtractor.setVocabulary(dictionary);
	for (int i = 0; i<listOfImages.size(); i++)
	{
		if (!openImage("train/" + listOfImages[i], image))
			continue;
		cout << "Computing from Image: " + listOfImages[i] + " of " + to_string(listOfImages.size()) << endl;

		detector->detect(image, keypoints);
		bowExtractor.compute(image, keypoints, descriptors);
		if (descriptors.cols != dictionarySize)
			emptydescriptors.push_back(i);
		bagofWords.push_back(descriptors);
	}
	read_csv("trainLabels.csv", label);
	applySVM(listOfTestImages, bowExtractor);
}

void applySVM(vector<string> &listOfTestImages, BOWImgDescriptorExtractor bowExtractor) {

	/*
	In practice, I suggest using SVM with different kernels before drawing the conclusion. Please try at least the following:
	Linear
	RBF
	Polynomial
	Chi-Squared
	*/

	cout << "applying Linear SVM" << endl;
	CvSVMParams paramsLinear;
	paramsLinear.svm_type = CvSVM::C_SVC;
	paramsLinear.kernel_type = CvSVM::LINEAR;
	paramsLinear.term_crit = tc;

	CvSVM SVMLinear;
	cout << "svm.train" << endl;
	SVMLinear.train(bagofWords, labels, Mat(), Mat(), paramsLinear);
	//SVM.save("svm.xml");

	cout << "applying RBF SVM" << endl;
	CvSVMParams paramsRBF;
	paramsRBF.svm_type = CvSVM::C_SVC;
	paramsRBF.kernel_type = CvSVM::RBF;
	paramsRBF.term_crit = tc;

	CvSVM SVMRBF;
	cout << "svm.train" << endl;
	SVMRBF.train(bagofWords, labels, Mat(), Mat(), paramsRBF);
	//SVMRBF.train_auto(bagofWords, labels, Mat(), Mat(),paramsRBF);
	//SVM.save("svm.xml");

	cout << "applying Polynomial SVM" << endl;
	CvSVMParams paramsPolynomial;
	paramsPolynomial.svm_type = CvSVM::C_SVC;
	paramsPolynomial.kernel_type = CvSVM::POLY;
	paramsPolynomial.degree = 3; // for poly
	paramsPolynomial.term_crit = tc;

	CvSVM SVMPolynomial;
	cout << "svm.train" << endl;
	SVMPolynomial.train(bagofWords, labels, Mat(), Mat(), paramsPolynomial);
	//SVM.save("svm.xml");


	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Mat image;
	Mat descriptors;
	std::vector<KeyPoint> keypoints;
	ofstream svmLinear("SVM LINEAR.csv");
	svmLinear << "id,label\n";
	ofstream svmRBF("SVM RBF.csv");
	svmRBF << "id,label\n";
	ofstream svmPoly("SVM POLINOMIAL.csv");
	svmPoly << "id,label\n";


	srand(time(NULL));
	for (int i = 0; i<listOfTestImages.size(); i++)
	{
		if (!openImage("test/" + listOfTestImages[i], image))
			continue;
		cout << "Computing from Test image: " + listOfTestImages[i] + " of " + to_string(listOfTestImages.size()) << endl;

		detector->detect(image, keypoints);
		bowExtractor.compute(image, keypoints, descriptors);
		if (descriptors.cols != dictionarySize) {
			svmLinear << i + 1 << "," << label[rand() % 10 ] << "\n";
			svmRBF << i + 1 << "," << label[rand() % 10] << "\n";
			svmPoly << i + 1 << "," << label[rand() % 10] << "\n";
			continue;
		}
		else {
			float resultLinear,resultRBF, resultPolinomial;
			resultLinear = SVMLinear.predict(descriptors);
			svmLinear << i + 1 << "," << label[resultLinear] << "\n";
			resultRBF = SVMRBF.predict(descriptors);
			svmRBF << i + 1 << "," << label[resultRBF] << "\n";
			resultPolinomial = SVMPolynomial.predict(descriptors);
			svmPoly << i + 1 << "," << label[resultPolinomial] << "\n";
		}
	}
}