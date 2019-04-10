// Copyright NVIDIA Corporation 2007 -- Ignacio Castano <icastano@nvidia.com>
// 
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
// 
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

// Math functions and operators to be used with vector types.

#ifndef CUDAMATHD_H
#define CUDAMATHD_H

#include <float.h>



inline __device__ __host__ double3 operator *(double3 a, double3 b)
{
    return make_double3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __device__ __host__ double4 operator *(const double4 a, const double4 b)
{
  return make_double4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

inline __device__ __host__ double3 operator *(double f, double3 v)
{
    return make_double3(v.x*f, v.y*f, v.z*f);
}

inline __device__ __host__ double4 operator *(const double4 v, const double f)
{
    return make_double4(v.x*f, v.y*f, v.z*f, v.w*f);
}
inline __device__ __host__ double4 operator *(const double f, const double4 v )
{
    return make_double4(v.x*f, v.y*f, v.z*f, v.w*f);
}

inline __device__ __host__ double3 operator *(double3 v, double f)
{
    return make_double3(v.x*f, v.y*f, v.z*f);
}

inline __device__ __host__ double4 operator -(double4 a, double4 b)
{
    return make_double4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

inline __device__ __host__ double4 operator +(double4 a, double4 b)
{
    return make_double4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

inline __device__ __host__ double3 operator +(double3 a, double3 b)
{
    return make_double3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ __host__ void operator +=(double3 & b, double3 a)
{
    b.x += a.x;
    b.y += a.y;
    b.z += a.z;
}

inline __device__ __host__ void operator +=(double4 & b, double4 a)
{
    b.x += a.x;
    b.y += a.y;
    b.z += a.z;
    b.w += a.w;
}

inline __device__ __host__ double3 operator -(double3 a, double3 b)
{
    return make_double3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ __host__ void operator -=(double3 & b, double3 a)
{
    b.x -= a.x;
    b.y -= a.y;
    b.z -= a.z;
}

inline __device__ __host__ double3 operator /(double3 v, double f)
{
    double inv = 1.0f / f;
    return v * inv;
}

inline __device__ __host__ void operator /=(double3 & b, double f)
{
    double inv = 1.0f / f;
    b.x *= inv;
    b.y *= inv;
    b.z *= inv;
}

inline __device__ __host__ bool operator ==(double3 a, double3 b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline __device__ __host__ double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ __host__ double dot(double4 a, double4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __device__ __host__ double clamp(double f, double a, double b)
{
    return max(a, min(f, b));
}

inline __device__ __host__ double3 clamp(double3 v, double a, double b)
{
    return make_double3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ double3 clamp(double3 v, double3 a, double3 b)
{
    return make_double3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}


inline __device__ __host__ double3 normalize(double3 v)
{
    double len = 1.0f / sqrtf(dot(v, v));
    return make_double3(v.x * len, v.y * len, v.z * len);
}




// Use power method to find the first eigenvector.
// http://www.miislita.com/information-retrieval-tutorial/matrix-tutorial-3-eigenvalues-eigenvectors.html
inline __device__ __host__ double3 firstEigenVector( double matrix[6] )
{
	// 8 iterations seems to be more than enough.

	double3 row0 = make_double3(matrix[0], matrix[1], matrix[2]);
	double3 row1 = make_double3(matrix[1], matrix[3], matrix[4]);
	double3 row2 = make_double3(matrix[2], matrix[4], matrix[5]);

	double r0 = dot(row0, row0);
	double r1 = dot(row1, row1);
	double r2 = dot(row2, row2);

	double3 v;
	if (r0 > r1 && r0 > r2) v = row0;
	else if (r1 > r2) v = row1;
	else v = row2;

	//double3 v = make_double3(1.0f, 1.0f, 1.0f);
	for(int i = 0; i < 8; i++) {
		double x = v.x * matrix[0] + v.y * matrix[1] + v.z * matrix[2];
		double y = v.x * matrix[1] + v.y * matrix[3] + v.z * matrix[4];
		double z = v.x * matrix[2] + v.y * matrix[4] + v.z * matrix[5];
		double m = max(max(x, y), z);        
		double iv = 1.0f / m;
		if (m == 0.0f) iv = 0.0f;
		v = make_double3(x*iv, y*iv, z*iv);
	}

	return v;
}

inline __device__ bool singleColor(const double3 * colors)
{
#if __DEVICE_EMULATION__
	bool sameColor = false;
	for (int i = 0; i < 16; i++)
	{
		sameColor &= (colors[i] == colors[0]);
	}
	return sameColor;
#else
	__shared__ int sameColor[16];
	
	const int idx = threadIdx.x;
	
	sameColor[idx] = (colors[idx] == colors[0]);
	sameColor[idx] &= sameColor[idx^8];
	sameColor[idx] &= sameColor[idx^4];
	sameColor[idx] &= sameColor[idx^2];
	sameColor[idx] &= sameColor[idx^1];
	
	return sameColor[0];
#endif
}

inline __device__ void colorSums(const double3 * colors, double3 * sums)
{
#if __DEVICE_EMULATION__
	double3 color_sum = make_double3(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < 16; i++)
	{
		color_sum += colors[i];
	}

	for (int i = 0; i < 16; i++)
	{
		sums[i] = color_sum;
	}
#else

	const int idx = threadIdx.x;

	sums[idx] = colors[idx];
	sums[idx] += sums[idx^8];
	sums[idx] += sums[idx^4];
	sums[idx] += sums[idx^2];
	sums[idx] += sums[idx^1];

#endif
}

inline __device__ double3 bestFitLine(const double3 * colors, double3 color_sum, double3 colorMetric)
{
	// Compute covariance matrix of the given colors.
#if __DEVICE_EMULATION__
	double covariance[6] = {0, 0, 0, 0, 0, 0};
	for (int i = 0; i < 16; i++)
	{
		double3 a = (colors[i] - color_sum * (1.0f / 16.0f)) * colorMetric;
		covariance[0] += a.x * a.x;
		covariance[1] += a.x * a.y;
		covariance[2] += a.x * a.z;
		covariance[3] += a.y * a.y;
		covariance[4] += a.y * a.z;
		covariance[5] += a.z * a.z;
	}
#else

	const int idx = threadIdx.x;

	double3 diff = (colors[idx] - color_sum * (1.0f / 16.0f)) * colorMetric;

	// @@ Eliminate two-way bank conflicts here.
	// @@ It seems that doing that and unrolling the reduction doesn't help...
	__shared__ double covariance[16*6];

	covariance[6 * idx + 0] = diff.x * diff.x;    // 0, 6, 12, 2, 8, 14, 4, 10, 0
	covariance[6 * idx + 1] = diff.x * diff.y;
	covariance[6 * idx + 2] = diff.x * diff.z;
	covariance[6 * idx + 3] = diff.y * diff.y;
	covariance[6 * idx + 4] = diff.y * diff.z;
	covariance[6 * idx + 5] = diff.z * diff.z;

	for(int d = 8; d > 0; d >>= 1)
	{
		if (idx < d)
		{
			covariance[6 * idx + 0] += covariance[6 * (idx+d) + 0];
			covariance[6 * idx + 1] += covariance[6 * (idx+d) + 1];
			covariance[6 * idx + 2] += covariance[6 * (idx+d) + 2];
			covariance[6 * idx + 3] += covariance[6 * (idx+d) + 3];
			covariance[6 * idx + 4] += covariance[6 * (idx+d) + 4];
			covariance[6 * idx + 5] += covariance[6 * (idx+d) + 5];
		}
	}

#endif

	// Compute first eigen vector.
	return firstEigenVector(covariance);
}


#endif // CUDAMATH_H
