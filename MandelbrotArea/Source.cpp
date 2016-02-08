#include <random>
#include <cstdint>
#include <algorithm>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

using namespace std;

int isInside(float * const  __restrict cx, float * const  __restrict cy) {
	const __m256 c_re = _mm256_load_ps(cx);
	const __m256 c_im = _mm256_load_ps(cy);

	const __m256 d_vec = _mm256_set_ps(2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f);
	const __m256 q_vec = _mm256_set_ps(4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f);

	__m256 result = _mm256_set_ps(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);

	__m256 x = _mm256_setzero_ps(), y = _mm256_setzero_ps();
	int resMask = 0;

	for (int c = 0; c < 400 / 8; ++c) {
		const __m256 fmad_x_c_re = _mm256_fmadd_ps(x, x, c_re);
		const __m256 m_yy = _mm256_mul_ps(y, y);
		__m256 newX = _mm256_sub_ps(fmad_x_c_re, m_yy);
		// float x_new = x*x - y*y + c_re;

		const __m256 m_xy = _mm256_mul_ps(x, y);
		const __m256 m_m_xy_d_vec = _mm256_mul_ps(m_xy, d_vec);
		y = _mm256_add_ps(c_im, m_m_xy_d_vec);
		// y = 2*x*y + c_im;

		x = newX;

		const __m256 m_yy2 = _mm256_mul_ps(y, y);
		const __m256 fmad_x_m_yy2 = _mm256_fmadd_ps(x, x, m_yy2);

		const __m256 out = _mm256_cmp_ps(fmad_x_m_yy2, q_vec, _CMP_GT_OS);
		result = _mm256_or_ps(result, out);
		//if (x*x + y*y > 4.0)

		resMask = _mm256_movemask_ps(result);

		if (resMask == (1 << 8) - 1) {
			return resMask;
		}
	}

	return resMask;
}

int main() {
	const float CxMin=-2.5;
	const float CxMax=1.5;
	const float CyMin=-2.0;
	const float CyMax=2.0;

	const float width = CxMax - CxMin;
	const float height = CyMax - CyMin;

	random_device dev;
	default_random_engine eng(dev());

	uniform_real_distribution<float> widthDist(CxMin, CxMax);
	uniform_real_distribution<float> heightDist(CyMin, CyMax);

	const double totalArea = width * height;

	uint64_t in_out[2] = {0, 0};
	float area = 0;

	double PCFreq = 0.0;
	__int64 CounterStart = 0;

	LARGE_INTEGER fr, start, end;
	BOOL res = QueryPerformanceFrequency(&fr);
	PCFreq = double(fr.QuadPart) / 1000.0;

	double total = 1e99;

	for (;;) {
		for (uint64_t c = 0; c < 1 << 10; ++c) {
			float xs[8], ys[8];

			for (int r = 0; r < 8; ++r) {
				xs[r] = widthDist(eng);
				ys[r] = heightDist(eng);
			}

			int res = isInside(xs, ys);

			for (int r = 0; r < 8; ++r) {
				in_out[(res >> r) & 0x1]++;
			}
		}

		const float abefore = area;
		area = double(area) * 0.5 + (totalArea * (double(in_out[0]) / double(in_out[1]))) * 0.5;

		if (std::fabs(abefore - area) < 1e-4) {
			break;
		}

		in_out[0] = in_out[1] = 0;
	}

	printf("\nTotal: %f\n", area);
	getchar();
	return 0;
}
