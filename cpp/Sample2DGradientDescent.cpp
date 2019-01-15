
/* Stream */
#include <fstream>
#include <iostream>

/* container */
#include <string>
#include <deque>


// ========================================================================================
// データ変換処理
// 読み込んだ1行のデータをカンマで区切って，doubleに変換処理したものをstd::deque<>に詰める
// 
// return 変換後のデータリスト
// 
// data：変換したいデータ
//
//
std::deque<double> parseLine(std::string data) {
	std::deque<double> result;
	std::string::size_type pos;
	
	while ((pos = data.find_first_of(','))!=std::string::npos ) {
		result.push_back(std::stod(data.substr(0, pos)));
		data = data.substr(pos + 1);
	}
	if (!data.empty()) {
		result.push_back(std::stod(data.substr(pos + 1)));
	}
	return result;
}

// ========================================================================================
// ファイル読み込み
// 
// return 読み込まれた学習データのリスト
//
// path：ファイルパス
//
//
std::deque<std::deque<double>> loadFile(const char *path) {
	std::deque<std::deque<double>> result;

	std::ifstream ifs(path);
	if (ifs.fail()) {
		return result;
	}
	
	std::string line;
	while (std::getline(ifs, line)) {
		result.push_back(parseLine(line));
	}
	return result;
}

// ========================================================================================
// コスト計算
// 
// return コスト
//
// X：入力データ
// y：教師データ
// theta：シータ値
//
//
double computeCost(
	const std::deque<std::deque<double>> &X,
	const std::deque<double> &y,
	const std::deque<double> &theta
) {
	int m = X.size();
	int n = theta.size();
	double J = 0.0;
	
	for (int i = 0; i < m; i++) {
		double cost = 0.0;
		const std::deque<double> &x = X[i];
		for (int j = 0; j < n; j++) {
			cost += x[j] * theta[j];
		}
		J += (cost - y[i]) * (cost - y[i]);
	}
	return J / (2.0 * m);
}


// ========================================================================================
// 勾配降下法の実行
//
// return 最適化されたシータ値
//
// X：入力データ
// y：教師データ
// theta：初期シータ値
// alpha：学習率（デフォルト0.01）
// iterations：反復数（デフォルト1500）
// 
//
std::deque<double> gradientDescent(
	const std::deque<std::deque<double>> &X,
	const std::deque<double> &y,
	std::deque<double> theta,
	double alpha = 0.01,
	int iterations = 1500
) {
	int m = y.size();
	int n = theta.size();
	int i, j;

	double val, K = alpha / m;
	double *grad = new double[n];
	std::deque<double> x;

	do {
		// grad initialize with zero
		memset(grad, 0, sizeof(double)*n);

		// grad (dJ/dt) の計算
		for (i = 0; i < m; i++) {
			x = X[i];
			val = -1.0 * y[i];
			for (j = 0; j < n; j++) {
				val += x[j] * theta[j];
			}
			for (j = 0; j < n; j++) {
				grad[j] += x[j] * val;
			}
		}

		for (j = 0; j < n; j++) {
			theta[j] -= K * grad[j];
		}
	} while (iterations-- > 0);
	delete[] grad;
	return theta;
}

// ========================================================================================
// エントリポイント
//
// argc：実行時引数argvの要素数
// argv：実行時引数のデータ（1つ目->自分自身のフルパス，2つ目以降->引数として指定したデータ）
//
//
int main(int argc, char* argv[]) {

	if (argc < 2) {
		std::cout << "need argument filepath" << std::endl;
		return 0;
	}

	std::deque<std::deque<double>> datalist = loadFile(argv[1]);
	if (datalist.empty()) {
		std::cout << "[ERROR] failure load file." << std::endl;
		return -1;
	}

#if defined(_DEBUG) || defined(DEBUG)
	for (std::deque<std::deque<double>>::size_type i = 0; i < datalist.size(); i++) {
		std::deque<double> ds = datalist[i];
		std::stringstream ss;
		for (int j = 0; j < 2 ; j++) {
			ss << ","<< ds[j];
		}
		ss << std::ends;
		std::cout << ss.str().substr(1) << std::endl;
	}
#endif

	std::cout << "---------------------------------------------------------------" << std::endl;
	std::cout << "dataset size: " << datalist.size() << std::endl;


	// == create dataset ====================================================
	std::deque<std::deque<double>> X;
	std::deque<double> y;
	for (std::deque<std::deque<double>>::size_type i = 0; i < datalist.size(); i++) {
		std::deque<double> ds = datalist[i];
		X.push_back(std::deque<double>({1.0, ds[0]}));
		y.push_back(ds[1]);
	}


	// == Compute cost value ====================================================
	std::cout << std::endl << std::endl
		<< "---------------------------------------------------------------" << std::endl;
	std::cout << "theta=[ 0, 0] -> " << computeCost(X, y, std::deque<double>({ 0.0, 0.0})) << std::endl;
	std::cout << "theta=[-1, 2] -> " << computeCost(X, y, std::deque<double>({-1.0, 2.0})) << std::endl;


	// == Gradient Descent ====================================================
	int iterations = 1500;
	double alpha = 0.01;

	std::deque<double> theta = gradientDescent(X, y, std::deque<double>({0.0, 0.0}), alpha, iterations);
	
	std::cout << std::endl << std::endl
		<< "---------------------------------------------------------------" << std::endl;
	std::cout << "Optimized parameter theta using gradient descent" << std::endl;
	std::cout << "theta = [" << theta[0] << ", " << theta[1] << "]" << std::endl;


	// == Predict ====================================================
	std::cout << std::endl << std::endl
		<< "---------------------------------------------------------------" << std::endl;
	double test[6] = { 3.5, 7.0, 10.0, 12.0, 15.0, 20.0 };
	for (int i = 0; i < 6; i++) {
		double x[] = { 1.0, test[i] };
		double predict = 0.0;
		for (int j = 0; j < 2; j++) {
			predict += x[j] * theta[j];
		}
		std::cout << "For population = " << test[i] * 10000 <<
			", predict a profit of " << predict << std::endl;
	}

	return 0;
}
