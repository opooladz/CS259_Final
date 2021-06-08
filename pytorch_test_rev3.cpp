#include <torch/torch.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <assert.h>

using namespace std;

torch::Tensor dct(const torch::Tensor x, string norm="None") { 
	/*
	Type II DCT
	Returns a 2D Tensor. If x is a N-sequence, the resulting tensor is of the shape (1,N)
	If x is a (N,N) matrix, the resulting tensor is also a (N,N) matrix
	*/

	// Get size of tensor x and reshape
	auto shape = x.sizes();
	int N = int(shape.back());
	int nrow;
	torch::Tensor x_;
	if (shape.size() == 1) {
		x_ = x.unsqueeze(0);
		nrow = 1;
	}	
	else if(shape.size() == 2){
		x_ = x;
		nrow = int(*(shape.begin()));
	}
	else {
		throw std::string("Not a 2D Tensor.");
	}

	// Declare k = 0, 1, 2, ... ,N-1
	torch::Tensor k = torch::arange(N);

	// construct y
	torch::Tensor y = torch::zeros_like(x_);
	for (int i = 0; i < nrow; i++) {
		for (int n = 0; n < N; n += 2) {
			y[i][n / 2] = x_[i][n].item();
		}
		for (int n = 0; n < N / 2; n++) {
			y[i][N - 1 - n] = x_[i][2 * n + 1].item();
		}
	}

	// Construct e^(jkpi/2N)
	auto cos = torch::cos(k * M_PI / (2 * N));
	auto sin = torch::sin(-1 * k * M_PI / (2 * N));
	auto e = torch::complex(cos, sin);
	
	// FFT of y
	torch::Tensor Y = torch::fft::fft(y);

	// DCT of x
	torch::Tensor X = torch::real(2 * torch::mul(Y, e));
	
	if (norm == "ortho") {
		for (int i = 0; i < nrow; i++) {
			X[i][0] /= sqrt(4 * N);
			for (int k = 1; k < N; k++) {
				X[i][k] /= sqrt(2 * N);
			}
		}
	}
	else if(norm != "None")
		cout << "Unrecognized input. Input ignored." << endl;

	return X;
}

torch::Tensor dct2(torch::Tensor x, string norm = "None") {
	torch::Tensor X1 = dct(x, norm);
	torch::Tensor X2 = dct(X1.transpose(1, 0), norm).transpose(1, 0);
	return X2;
}


auto main() -> int {
	const int N = 9;
	
	torch::Tensor x = torch::arange(N);
	torch::Tensor x2 = torch::cat({ x.reshape({ 1, N }),
		x.reshape({ 1, N }),
		x.reshape({ 1, N }), 
		x.reshape({ 1, N }),
		x.reshape({ 1, N }),
		x.reshape({ 1, N }),
		x.reshape({ 1, N }),
		x.reshape({ 1, N }),
		x.reshape({ 1, N }) }, 0);
	torch::Tensor X;
	torch::Tensor X2;

	// 1D sequence, no normalization
	X = dct(x);
	cout << X.squeeze() << endl;

	// 1D sequence, with normalization
	X = dct(x, "ortho");
	cout << X.squeeze() << endl;

	// 2D matrix, no normalization
	X2 = dct2(x2).squeeze();
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << X2[i][j].item() << "\t";
		}
		cout << endl;
	}

	// 2D matrix, with normalization
	X2 = dct2(x2, "ortho").squeeze();
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << X2[i][j].item() << "\t";
		}
		cout << endl;
	}

}
