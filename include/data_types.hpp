#include <limits.h>
#include <stdio.h>

#ifdef __OPEN_MP__
#pragma omp declare target 
#else
#define __HD__ __host__ __device__ 
#endif

struct dim4{
  double x, y, z, w;

  __HD__ dim4(): x(0.0), y(0.0), z(0.0), w(0.0) {  }

  template<class P>
  __HD__ dim4(P val) {
    x = static_cast<double>(val);
    y = static_cast<double>(val);
    z = static_cast<double>(val);
    w = static_cast<double>(val);
  }

  __HD__ dim4 (double vx, double vy, double vz, double vw) : x(vx), y(vy), z(vz), w(vw) {}

  template <class M>
  __HD__ dim4& operator=(const M& a)
  {
    x=a;
    y=a;
    z=a;
    w=a;
    return *this;
  }

  __HD__ dim4 operator+(const dim4& a) const{
    return dim4(a.x+x, a.y+y, a.z+z, a.w+w);
  }

  __HD__ dim4 operator-(const dim4& a) const{
    return dim4(x - a.x, y - a.y, z  - a.z, w - a.w);
  }

  __HD__ dim4 operator*(const dim4& a) const{
    return dim4(a.x*x, a.y*y, a.z*z, a.w*w);
  }

  __HD__ dim4 operator/(const dim4& a) const{
    return dim4(x/a.x, y/a.y, z/a.z, w/a.w);
  }

  template <class M>
  __HD__ dim4 operator/(const M a) const{
    return dim4(x/a, y/a, z/a, w/a);
  }

  __HD__ friend bool operator == (const dim4& lhs, long rhs){
    if ( lhs.x == lhs.y == lhs.z == lhs.w == static_cast<double>(rhs))
      return true;
    return false;
  }

  __HD__ friend bool operator == (const dim4& lhs, double rhs){
    if ( lhs.x == lhs.y == lhs.z == lhs.w == rhs)
      return true;
    return false;
  }

  __HD__ friend bool operator == (const dim4& lhs, const dim4& rhs){
    if ( lhs.x == rhs.x && lhs.y ==  rhs.y && lhs.z == rhs.z && lhs.w == rhs.w)
      return true;
    return false;
  }

  __HD__ friend bool operator > ( const dim4& lhs, const dim4 rhs){
    double val = (lhs.x - rhs.x) +
              (lhs.y - rhs.y) +
              (lhs.z - rhs.z) +
              (lhs.w - rhs.w); 
    return (val > 0.0);
  }

  __HD__ friend bool operator < ( const dim4& lhs, const dim4 rhs){
    double val = (lhs.x - rhs.x) +
              (lhs.y - rhs.y) +
              (lhs.z - rhs.z) +
              (lhs.w - rhs.w); 
    return (val < 0.0);
  }
};

#define SIZE 32
struct dim32{
  double x[SIZE];

  __HD__ dim32() {
    for ( int i = 0; i < SIZE; i++)
      x[i] = 0.0;
  }

  template<class P>
  __HD__ dim32(P val) {
    for ( int i = 0 ; i < SIZE; i++)
      x[i] = static_cast<double>(val);
   }

  __HD__ dim32 (double *v) {
    for ( int i = 0 ; i < SIZE; i++)
      x[i] = v[i];
  }

  template <class M>
  __HD__ dim32& operator=(const M a)
  {
    for ( int i = 0 ; i < SIZE; i++)
      x[i] = static_cast<double>(a);
    return *this;
  }


  template <class M>
  __HD__ dim32& operator=(const M& a)
  {
    for ( int i = 0 ; i < SIZE; i++)
      x[i] = static_cast<double>(a);
    return *this;
  }

  __HD__ dim32 operator+(const dim32& a) const{
    double tmp[SIZE];
    for ( int i = 0; i < SIZE; i++){
      tmp[i] = x[i] + a.x[i];
    }
    return dim32(tmp);
  }

  __HD__ dim32 operator-(const dim32& a) const{
    double tmp[SIZE];
    for ( int i = 0; i < SIZE; i++){
      tmp[i] = x[i] - a.x[i];
    }
    return dim32(tmp);
  }

  __HD__ dim32 operator*(const dim32& a) const{
    double tmp[SIZE];
    for (int i = 0; i < SIZE; i++){
      tmp[i] = x[i] * a.x[i];
    }
    return dim32(tmp);
  }

  __HD__ dim32 operator/(const dim32& a) const{
    double tmp[SIZE];
    for ( int i = 0; i < SIZE; i++){
      tmp[i] = x[i] / a.x[i];
    }
    return dim32(tmp);
  }

  template <class M>
  __HD__ dim32 operator/(const M a) const{
    double tmp[SIZE];
    for (int i = 0; i < SIZE; i++){
      tmp[i] = x[i] / a;
    }
    return dim32(tmp);
  }

  __HD__ friend bool operator == (const dim32& lhs, const dim32& rhs){
    for (int i = 0 ; i < SIZE; i++){
      if ( lhs.x[i] != rhs.x[i] )
        return false; 
    }
    return true;
  }

  __HD__ friend bool operator == (const dim32& lhs, double rhs){
    for (int i = 0 ; i < SIZE; i++){
      if ( lhs.x[i] != rhs )
        return false; 
    }
    return true;
  }

  __HD__ friend bool operator > ( const dim32& lhs, const dim32 rhs){
    double val = 0.0;
    for ( int i = 0; i < SIZE; i++){ 
      val += lhs.x[i] - rhs.x[i];
    }
    return (val > 0.0);
  }

  __HD__ friend bool operator < ( const dim32& lhs, const dim32 rhs){
    double val = 0.0;
    for ( int i = 0; i < SIZE; i++){ 
      val += lhs.x[i] - rhs.x[i];
    }
    return (val < 0.0);
  }
};


#ifdef __OPEN_MP__
#pragma omp end declare target 

#pragma omp declare reduction(min : dim4: \
            omp_out = omp_in > omp_out ? omp_out : omp_in )\
 initializer( omp_priv = { std::numeric_limits<double>::max() } )

#pragma omp declare reduction(max: dim4: \
        omp_out = omp_in < omp_out ? omp_out : omp_in )\
 initializer( omp_priv = {  std::numeric_limits<double>::min() } )

#pragma omp declare reduction(*: dim4:\
  omp_out = omp_in*omp_out) \
  initializer (omp_priv=1.0)

#pragma omp declare reduction(+: dim4:\
  omp_out = omp_in+omp_out) \
  initializer (omp_priv=0.0)

#pragma omp declare reduction(min : dim32: \
            omp_out = omp_in > omp_out ? omp_out : omp_in )\
 initializer( omp_priv = dim32(std::numeric_limits<double>::max())  )

#pragma omp declare reduction(max: dim32: \
        omp_out = omp_in < omp_out ? omp_out : omp_in )\
 initializer( omp_priv = dim32(std::numeric_limits<double>::min()) )


#pragma omp declare reduction(*: dim32:\
  omp_out = omp_in*omp_out) \
  initializer (omp_priv=1.0)

#pragma omp declare reduction(+: dim32:\
  omp_out = omp_in+omp_out) \
  initializer (omp_priv=0.0)
#endif
