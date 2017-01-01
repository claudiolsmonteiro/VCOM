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

// auxiliary functions


bool openImage(const std::string &filename, Mat &image);
void train(vector<string> &listOfImages);
void bagOfWords(vector<string> &listOfImages, vector<string> &listOfTestImages);
void read_csv(const string& filename, vector<string> label);
//void classifySVM(vector<string> &listOfTestImages, BOWImgDescriptorExtractor bowExtractor);
//void classifyKNN(vector<string> &listOfTestImages, BOWImgDescriptorExtractor bowExtractor);
void classify(vector<string> &listOfTestImages, BOWImgDescriptorExtractor bowExtractor);

//Global variables
Mat allTrainDescriptors, bagofWords, labels;
//the number of bags
int dictionarySize = 200;
//define Term Criteria
TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
//vector with the training elements that will be ignored for having 0 Descriptors
vector<int> emptydescriptors;
vector<string> label;
int  choice = 0, choiceSVM = 0;


int main(int argc, char** argv)
{
    initModule_nonfree();

    ///////////////////////////////////////
    cout << "Both training methods [0] , KNN [1] , SVM [2] ";
    string line;
    getline(cin, line);
    choice = atoi(line.c_str());
    cout << "The value you entered is " << choice << "\n\n";
    
    // Cycle to chose the methods to be used to train and predict
    while (choice < 0 || choice > 2) // Check for invalid input
    {
        cout << "Wrong choice, try again." << endl;
        cout << "Both training methods [0] , KNN [1] , SVM [2] ";
        getline(cin, line);
        choice = atoi(line.c_str());
        cout << "The value you entered is " << choice << endl;
    }
    if(choice == 0 || choice == 2) {
        cout << "Every kernel [0] , Linear [1] , RBF [2], Polynomial [3] ";
        getline(cin, line);
        choiceSVM = atoi(line.c_str());
        cout << "The value you entered is " << choiceSVM << "\n\n";
        while (choiceSVM < 0 || choiceSVM > 3) // Check for invalid input
        {
            cout << "Wrong Kernel choice, try again." << endl;
            cout << "Every kernel [0] , Linear [1] , RBF [2], Polynomial [3] ";
            getline(cin, line);
            choiceSVM = atoi(line.c_str());
            cout << "The value you entered is " << choiceSVM << endl;
        }
    }
    //create vectors with 50000 training images and 290000 junk images + 10000 test images
    vector<string> listOfImages;
    for (int i = 0; i < 50000; i++)
        listOfImages.push_back(to_string(i+1) + ".png");
    vector<string> listOfTestImages;
    for (int i = 0; i < 300000; i++)
        listOfTestImages.push_back(to_string(i + 1) + ".png");
    
    train(listOfImages);
    //fill the label vectors with the 10 categories
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
    
    // create the bag of words followed by the classification
    bagOfWords(listOfImages, listOfTestImages);
    
    
    //applySVM(listOfTestImages);
    
    waitKey(0);
    return 0;
}
// function that reads the csv with the training labels and stores in a new vector their position
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
            // if the current label belongs to a image with 0 descriptors ignore it, else add it to the correct vector
            continue;
        }
        else {
            int pos = find(label.begin(), label.end(), classlabel) - label.begin();
            //cout << pos << endl;
            labels.push_back(pos);
        }
    }
}
// function used to open an image in grayscale
bool openImage(const std::string &filename, Mat &image)
{
    image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data) {
        std::cout << " --(!) Error reading image " << filename << std::endl;
        return false;
    }
    return true;
}
// function that uses SIFT to detect and extract keypoints pushing the results to a vector with the allTrainDescriptors vector
void train(vector<string> &listOfImages) {
    Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
    //Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
    //Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
    Mat image;
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    // for cycle to extract from all of the 50k images from the train folder
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
// function to create the bag of words using Flann as a matcher
void bagOfWords(vector<string> &listOfImages, vector<string> &listOfTestImages) {
    cout << " Creating Bag of Words" << endl;
    Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
    //Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
    //Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    Mat image;
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    
    //
    BOWKMeansTrainer bowTrainer(dictionarySize, tc, 1, KMEANS_PP_CENTERS);
    BOWImgDescriptorExtractor bowExtractor(extractor, matcher);
    Mat dictionary = bowTrainer.cluster(allTrainDescriptors);
    bowExtractor.setVocabulary(dictionary);
    // for cycle with the 50k training images to extract the actual bag of word descriptors
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
    // call of the read_csv function so it can ignore the images that have 0 descriptors
    read_csv("trainLabels.csv", label);
    // function that will use both/either SVM or KNN
    classify(listOfTestImages,bowExtractor);
    //applySVM(listOfTestImages, bowExtractor);
    //applyKNN(listOfTestImages, bowExtractor);
}

void classify(vector<string> &listOfTestImages,BOWImgDescriptorExtractor bowExtractor) {
    /*
     In practice, I suggest using SVM with different kernels before drawing the conclusion. Please try at least the following:
     Linear
     RBF
     Polynomial
     */
    cout << "About to classify, declaring parameters " << endl;
    Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
    
    // definition of Linear Svm parameters
    CvSVMParams paramsLinear;
    paramsLinear.svm_type = CvSVM::C_SVC;
    paramsLinear.kernel_type = CvSVM::LINEAR;
    paramsLinear.term_crit = tc;
    
    CvSVM SVMLinear;
    
    //SVM.save("svm.xml");
    
    // definition of Linear RBF parameters
    CvSVMParams paramsRBF;
    paramsRBF.svm_type = CvSVM::C_SVC;
    paramsRBF.kernel_type = CvSVM::RBF;
    paramsRBF.term_crit = tc;
    
    CvSVM SVMRBF;
    
    //SVM.save("svm.xml");
    
    // definition of Polynomial Svm parameters
    CvSVMParams paramsPolynomial;
    paramsPolynomial.svm_type = CvSVM::C_SVC;
    paramsPolynomial.kernel_type = CvSVM::POLY;
    paramsPolynomial.degree = 2; // mandatory for Polynomial
    paramsPolynomial.term_crit = tc;
    
    CvSVM SVMPolynomial;
    
    //SVM.save("svm.xml");
    const int K = 10;
    CvKNearest knn;
    //.csv files needed for validation on the kagle website
    ofstream kNN("KNN "+ to_string(dictionarySize) +" K "+ to_string(K)+".csv");
    ofstream svmLinear("SVM LINEAR "+ to_string(dictionarySize)+".csv");
    ofstream svmRBF("SVM RBF "+ to_string(dictionarySize)+".csv");
    ofstream svmPoly("SVM POLYNOMIAL "+ to_string(dictionarySize)+".csv");

    //svm/knn training depending on the user previous choices
    if ( choice == 0 || choice == 1) {
        cout << "applying Knn" << endl;
        kNN << "id,label" << endl;
        cout << "knn.train" << endl;
        knn.train(bagofWords, labels, Mat());
    }
    if( choice == 0 || choice == 2) {
        if( choiceSVM == 0 || choiceSVM == 1) {
            cout << "applying Linear SVM" << endl;
            cout << "svm.train" << endl;
            SVMLinear.train(bagofWords, labels, Mat(), Mat(), paramsLinear);
            svmLinear << "id,label\n";
        }
        if( choiceSVM == 0 || choiceSVM == 2) {
            cout << "applying RBF SVM" << endl;
            cout << "svm.train" << endl;
            SVMRBF.train(bagofWords, labels, Mat(), Mat(), paramsRBF);
            //SVMRBF.train_auto(bagofWords, labels, Mat(), Mat(),paramsRBF);
            svmRBF << "id,label\n";
        }
        if( choiceSVM == 0 || choiceSVM == 3) {
            cout << "applying Polynomial SVM" << endl;
            cout << "svm.train" << endl;
            SVMPolynomial.train(bagofWords, labels, Mat(), Mat(), paramsPolynomial);
            svmPoly << "id,label\n";
        }
    }
    Mat image;
    Mat descriptors;
    vector<KeyPoint> keypoints;

    
    //srand(time(NULL));
    
    //calculation of the test images
    for (int i = 0; i<listOfTestImages.size(); i++)
    {
        if (!openImage("test/" + listOfTestImages[i], image))
            continue;
        cout << "Computing from Test image: " + listOfTestImages[i] + " of " + to_string(listOfTestImages.size()) << endl;
        
        detector->detect(image, keypoints);
        bowExtractor.compute(image, keypoints, descriptors);
        // if it has no descriptors assign the rest of the division from the iterator with the number of labels
        if (descriptors.cols != dictionarySize) {
            if ( choice == 0 || choice == 1) {
                kNN << i + 1 << "," << label[i % 10] << "\n";
            }
            if( choice == 0 || choice == 2) {
                if( choiceSVM == 0 || choiceSVM == 1) {
                    svmLinear << i + 1 << "," << label[i % 10 ] << "\n";
                }
                if( choiceSVM == 0 || choiceSVM == 2) {
                    svmRBF << i + 1 << "," << label[i % 10] << "\n";
                }
                if( choiceSVM == 0 || choiceSVM == 3) {
                    svmPoly << i + 1 << "," << label[i % 10] << "\n";
                }
            }
        }
        else {
            // find/ predict the correct label and store the image number + "," + the predicted label
            float resultKnn, resultLinear,resultRBF, resultPolinomial;
            if ( choice == 0 || choice == 1) {
                resultKnn = knn.find_nearest(descriptors,K);
                kNN << i + 1 << "," << label[resultKnn] << "\n";
            }
            if( choice == 0 || choice == 2) {
                if( choiceSVM == 0 || choiceSVM == 1) {
                    resultLinear = SVMLinear.predict(descriptors);
                    svmLinear << i + 1 << "," << label[resultLinear] << "\n";
                }
                if( choiceSVM == 0 || choiceSVM == 2) {
                    resultRBF = SVMRBF.predict(descriptors);
                    svmRBF << i + 1 << "," << label[resultRBF] << "\n";
                }
                if( choiceSVM == 0 || choiceSVM == 3) {
                    resultPolinomial = SVMPolynomial.predict(descriptors);
                    svmPoly << i + 1 << "," << label[resultPolinomial] << "\n";
                }
            }
        }
    }
}
