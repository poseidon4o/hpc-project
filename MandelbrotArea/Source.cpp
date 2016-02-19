#include <random>
#include <cstdint>
#include <algorithm>
#include <immintrin.h>

using namespace std;

#if __linux__ != 0
#include <time.h>

static uint64_t timer_nsec() {
#if defined(CLOCK_MONOTONIC_RAW)
	const clockid_t clockid = CLOCK_MONOTONIC_RAW;

#else
	const clockid_t clockid = CLOCK_MONOTONIC;

#endif

	timespec t;
	clock_gettime(clockid, &t);

	return t.tv_sec * 1000000000UL + t.tv_nsec;
}

#elif _WIN64 != 0
#define NOMINMAX
#include <Windows.h>

static uint64_t timer_nsec() {

	static LARGE_INTEGER freq;
	static BOOL once = QueryPerformanceFrequency(&freq);

	LARGE_INTEGER t;
	QueryPerformanceCounter(&t);

	return 1000000000ULL * t.QuadPart / freq.QuadPart;
}

#elif __APPLE__ != 0
#include <mach/mach_time.h>

static uint64_t timer_nsec() {

    static mach_timebase_info_data_t tb;
    if (0 == tb.denom)
		mach_timebase_info(&tb);

    const uint64_t t = mach_absolute_time();

    return t * tb.numer / tb.denom;
}

#endif


#define FMA_ENABLED
#define SSE_RAND

using namespace std;

const int maxIter = 4000;
const double expectedArea = 1.5065918849;

int isInsideD(double * const __restrict cx, double * const __restrict cy) {
	const __m256d c_re = _mm256_load_pd(cx);
	const __m256d c_im = _mm256_load_pd(cy);

	const __m256d d_vec = _mm256_set_pd(2., 2., 2., 2.);
	const __m256d q_vec = _mm256_set_pd(4., 4., 4., 4.);

	__m256d result = _mm256_set_pd(0., 0., 0., 0.);

	__m256d x = _mm256_setzero_pd(), y = _mm256_setzero_pd();
	int resMask = 0;

	for (int c = 0; c < maxIter; ++c) {
#ifdef FMA_ENABLED
		const __m256d fmad_x_c_re = _mm256_fmadd_pd(x, x, c_re);
#else
		const __m256d d_x = _mm256_mul_pd(x, x);
		const __m256d fmad_x_c_re = _mm256_add_pd(d_x, c_re);
#endif
		const __m256d m_yy = _mm256_mul_pd(y, y);
		__m256d newX = _mm256_sub_pd(fmad_x_c_re, m_yy);
		// float x_new = x*x - y*y + c_re;

		const __m256d m_xy = _mm256_mul_pd(x, y);
		const __m256d m_m_xy_d_vec = _mm256_mul_pd(m_xy, d_vec);
		y = _mm256_add_pd(c_im, m_m_xy_d_vec);
		// y = 2*x*y + c_im;

		x = newX;

		const __m256d m_yy2 = _mm256_mul_pd(y, y);
#ifdef FMA_ENABLED
		const __m256d fmad_x_m_yy2 = _mm256_fmadd_pd(x, x, m_yy2);
#else
		const __m256d d_x2 = _mm256_mul_pd(x, x);
		const __m256d fmad_x_m_yy2 = _mm256_add_pd(d_x2, m_yy2);
#endif

		const __m256d out = _mm256_cmp_pd(fmad_x_m_yy2, q_vec, _CMP_GT_OS);
		result = _mm256_or_pd(result, out);
		//if (x*x + y*y > 4.0)

		resMask = _mm256_movemask_pd(result);

		if (resMask == (1 << 4) - 1) {
			return resMask;
		}
	}

	return resMask;
}

int isInside(float * const  __restrict cx, float * const  __restrict cy) {
	const __m256 c_re = _mm256_load_ps(cx);
	const __m256 c_im = _mm256_load_ps(cy);

	const __m256 d_vec = _mm256_set_ps(2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f);
	const __m256 q_vec = _mm256_set_ps(4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f);

	__m256 result = _mm256_set_ps(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);

	__m256 x = _mm256_setzero_ps(), y = _mm256_setzero_ps();
	int resMask = 0;

	for (int c = 0; c < maxIter; ++c) {
#ifdef FMA_ENABLED
		const __m256 fmad_x_c_re = _mm256_fmadd_ps(x, x, c_re);
#else
		const __m256 d_x = _mm256_mul_ps(x, x);
		const __m256 fmad_x_c_re = _mm256_add_ps(d_x, c_re);
#endif
		const __m256 m_yy = _mm256_mul_ps(y, y);
		__m256 newX = _mm256_sub_ps(fmad_x_c_re, m_yy);
		// float x_new = x*x - y*y + c_re;

		const __m256 m_xy = _mm256_mul_ps(x, y);
		const __m256 m_m_xy_d_vec = _mm256_mul_ps(m_xy, d_vec);
		y = _mm256_add_ps(c_im, m_m_xy_d_vec);
		// y = 2*x*y + c_im;

		x = newX;

		const __m256 m_yy2 = _mm256_mul_ps(y, y);
#ifdef FMA_ENABLED
		const __m256 fmad_x_m_yy2 = _mm256_fmadd_ps(x, x, m_yy2);
#else
		const __m256 d_x2 = _mm256_mul_ps(x, x);
		const __m256 fmad_x_m_yy2 = _mm256_add_ps(d_x2, m_yy2);
#endif

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


__declspec(align(16)) static __m128i cur_seed;
void srand_sse(unsigned int seed) {
    cur_seed = _mm_set_epi32(seed, seed + 1, seed, seed + 1);
}

void rand_sse(double * result) {
    __declspec(align(16)) static const unsigned int mult[4] = { 214013, 17405, 214013, 69069 };
    __declspec(align(16)) static const unsigned int gadd[4] = { 2531011, 10395331, 13737667, 1 };
    __declspec(align(16)) static const unsigned int mask[4] = { 0xFFFFFFFF, 0, 0xFFFFFFFF, 0 };
    __declspec(align(16)) static const unsigned int masklo[4] ={ 0x00007FFF, 0x00007FFF, 0x00007FFF, 0x00007FFF };

    __declspec(align(32)) static const double top[4] = { -10., -10., -10., -10. };
    __declspec(align(32)) static const double bot[4] = { 10., 10., 10., 10. };

    __m256d d_res = _mm256_cvtepi32_pd(cur_seed);
    __declspec(align(32)) static const double devs[4] = { 20.0 / (double)UINT_MAX, 20.0 / (double)UINT_MAX, 20.0 / (double)UINT_MAX, 20.0 / (double)UINT_MAX };
    __declspec(align(32)) static const double shift[4] = { -5., -5., -5., -5. };

    int hi = 0, lo = 0;

    do {

        __declspec(align(16)) __m128i cur_seed_split;
        __declspec(align(16)) __m128i multiplier;
        __declspec(align(16)) __m128i adder;
        __declspec(align(16)) __m128i mod_mask;
        __declspec(align(16)) __m128i sra_mask;

        adder = _mm_load_si128((__m128i*) gadd);
        multiplier = _mm_load_si128((__m128i*) mult);
        mod_mask = _mm_load_si128((__m128i*) mask);
        sra_mask = _mm_load_si128((__m128i*) masklo);
        cur_seed_split = _mm_shuffle_epi32(cur_seed, _MM_SHUFFLE(2, 3, 0, 1));

        cur_seed = _mm_mul_epu32(cur_seed, multiplier);
        multiplier = _mm_shuffle_epi32(multiplier, _MM_SHUFFLE(2, 3, 0, 1));
        cur_seed_split = _mm_mul_epu32(cur_seed_split, multiplier);

        cur_seed = _mm_and_si128(cur_seed, mod_mask);
        cur_seed_split = _mm_and_si128(cur_seed_split, mod_mask);
        cur_seed_split = _mm_shuffle_epi32(cur_seed_split, _MM_SHUFFLE(2, 3, 0, 1));
        cur_seed = _mm_or_si128(cur_seed, cur_seed_split);
        cur_seed = _mm_add_epi32(cur_seed, adder);


        d_res = _mm256_fmadd_pd(d_res, _mm256_load_pd(devs), _mm256_load_pd(shift));
        hi = _mm256_movemask_pd(_mm256_cmp_pd(d_res, _mm256_load_pd(top), _CMP_LT_OQ));
        lo = _mm256_movemask_pd(_mm256_cmp_pd(d_res, _mm256_load_pd(bot), _CMP_GT_OQ));
    } while (hi != 0 || lo != 0);

    _mm256_store_pd(result, d_res);
}


int main() {
	const double CxMin=-10;
	const double CxMax=10;
	const double CyMin=-10.0;
	const double CyMax=10.0;

	const double width = CxMax - CxMin;
	const double height = CyMax - CyMin;

	random_device dev;
	default_random_engine eng(dev());

	uniform_real_distribution<double> widthDist(CxMin, CxMax);
	uniform_real_distribution<double> heightDist(CyMin, CyMax);

	const double totalArea = width * height;

	uint64_t in_out[2] = {0, 0};
	double area = 0;

	double total = 1e99;

	uint64_t t_start, t_end;
    srand_sse(42);

	for (;;) {

		t_start = timer_nsec();
		for (uint64_t c = 0; c < 1 << 20; ++c) {
            __declspec(align(32)) double xs[4], ys[4];

#ifdef SSE_RAND
            rand_sse(xs);
            rand_sse(ys);
#else
			for (int r = 0; r < 4; ++r) {
				xs[r] = widthDist(eng);
				ys[r] = heightDist(eng);
			}
#endif
			int res = isInsideD(xs, ys);

			for (int r = 0; r < 4; ++r) {
				in_out[(res >> r) & 0x1]++;
			}
		}
		t_end = timer_nsec();

		printf("batch time: %fms\n", double(t_end - t_start) * 1e-6);

		area = totalArea * (double(in_out[0]) / double(in_out[1]));
		printf("Total: %f dev %f in[%llu] out[%llu]\n", area, std::abs(expectedArea - area), in_out[1], in_out[0]);


		if (std::abs(expectedArea - area) < 1e-9) {
			break;
		}

		// in_out[0] = in_out[1] = 0;
	}

	printf("\nTotal: %f\n", area);
	getchar();
	return 0;
}
