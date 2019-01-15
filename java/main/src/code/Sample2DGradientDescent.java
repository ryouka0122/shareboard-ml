package code;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * 勾配降下法のサンプル
 * @author ryouka
 *
 */
public class Sample2DGradientDescent {
	
	/**
	 * [Entry Point]
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		if(args==null || args.length<1) {
			System.out.println("[ERROR] need filepath");
			return;
		}
				
		// == load data ===========================================
		List<double[]> dataList = new ArrayList<>();
		try( Stream<String> stream = Files.lines(Paths.get(args[0])) ) {
			stream.forEach(line -> {
				dataList.add(parseLine(line));
			});
		}
		
		int m = dataList.size();
		
		System.out.printf("\n\n--------------------------------------------------\n");
		System.out.println("dataset size: " + m);
		
		// == create dataset ====================================================
		double[][] X = dataList.stream()
				.map(data -> new double[] {1.0, data[0]})
				.toArray(double[][]::new);
		
		double[] y = dataList.stream()
				.mapToDouble(data -> data[1])
				.toArray();
		
		
		// == Compute cost value ====================================================
		System.out.printf("\n\n--------------------------------------------------\n");
		System.out.printf("theta=[0, 0] -> %f\n", computeCost(X, y, new double[] {0.0, 0.0}));
		System.out.printf("theta=[-1, 2] -> %f\n", computeCost(X, y, new double[] {-1.0, 2.0}));
		

		// == Gradient Descent ====================================================
		int iterations = 1500;
		double alpha = 0.01;
		
		double[] theta = gradientDescent(X, y, new double[] {0.0, 0.0}, alpha, iterations);
		System.out.printf("\n\n--------------------------------------------------\n");
		System.out.printf("Optimized parameter theta using gradient descent\n");
		System.out.printf("theta = [%f, %f]\n", theta[0], theta[1]);
		
		
		// == Predict ====================================================
		System.out.printf("\n\n--------------------------------------------------\n");
		
		DoubleStream.of(3.5, 7.0, 10.0, 12.0, 15.0, 20.0)
		.mapToObj(v-> new double[] {1.0, v})
		.forEach( input -> {
			double predict = 0.0;
			for(int j=0 ; j<theta.length; j++) {
				predict += input[j] * theta[j];
			}
			System.out.printf("For Population = %f, predict a profit of %f\n",
					input[1]*10000, predict);
		});
		
	}
	
	/**
	 * CSV形式のデータからdouble型の配列に変換する処理
	 * @param line
	 * @return
	 */
	private static double[] parseLine(String line) {
		return Arrays.stream(line.split(","))
				.mapToDouble(Double::parseDouble)
				.toArray();		
	}
	
	/**
	 * コストを計算するメソッド
	 * @param X 入力データ
	 * @param y 教師データ
	 * @param theta パラメータ
	 * @return コスト
	 */
	private static double computeCost(double[][] X, double[] y, double[] theta) {
		int m = y.length;
		int n = theta.length;
		double J = 0.0;
		for(int i=0 ; i<m ; i++) {
			double cost = 0.0;
			double[] x = X[i];
			// 回帰の時のコスト計算（誤差2乗）
			for(int j=0 ; j<n ; j++) {
				cost += x[j] * theta[j];
			}
			J += (cost - y[i]) * (cost - y[i]);
		}
		return J / (2.0 * m);
	}
	
	/**
	 * 勾配降下法の実行
	 * @param X 入力データ
	 * @param y 教師データ
	 * @param theta 初期パラメータ
	 * @param alpha 学習率
	 * @param iterations 反復数
	 * @return 最適化されたパラメータ
	 */
	private static double[] gradientDescent(
			double[][] X,
			double[] y,
			double[] theta,
			double alpha,
			int iterations
	) {
		int m = y.length;
		int n = theta.length;
		
		double K = alpha / m;
		// loop in iterations
		for(int iter=0 ; iter<iterations ; iter++) {
			double[] grad = new double[n];
			// loop for each dataset
			for(int i=0; i<m ; i++) {
				double[] x = X[i];
				double val = 0.0;
				for(int j=0 ; j<n ; j++) {
					val += x[j] * theta[j];
				}
				val -= y[i];
				for(int j=0 ; j<n ; j++) {
					grad[j] += x[j] * val;
				}
			}
			
			for(int j=0 ; j<n ; j++) {
				theta[j] = theta[j] - K * grad[j];
			}
		}
		return theta;
	}
	
}
